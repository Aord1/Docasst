<script setup>
import { computed, nextTick, ref, watch } from "vue";

const apiBase = import.meta.env.VITE_API_BASE || "";
const activeNav = ref("chat");
const chatInput = ref("");
const tenantId = ref("default");
const fileInput = ref(null);
const chatFileInput = ref(null);
const loading = ref(false);
const ingesting = ref(false);
const pendingFiles = ref([]);
const messagesContainer = ref(null);
const streamMeta = ref({
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
});

const STORAGE_KEY = "docasst_web_sessions_v1";
const sessions = ref([]);
const currentSessionId = ref("");

function makeSession() {
  const localId = `local-${Date.now()}`;
  return {
    id: localId,
    threadId: "",
    title: "新会话",
    messages: [{ role: "assistant", content: "已新建会话，请输入你的问题。" }],
    uploadedFiles: [],
    updatedAt: Date.now(),
  };
}

function createNewSession() {
  const session = makeSession();
  sessions.value.unshift(session);
  currentSessionId.value = session.id;
}

function selectSession(id) {
  currentSessionId.value = id;
}

const currentSession = computed(() => {
  return sessions.value.find((s) => s.id === currentSessionId.value) || null;
});
const currentMessages = computed(() => currentSession.value?.messages || []);

function ensureCurrentSession() {
  if (!currentSession.value) {
    createNewSession();
  }
  return currentSession.value;
}

async function scrollToBottom(smooth = true) {
  await nextTick();
  const el = messagesContainer.value;
  if (!el) return;
  el.scrollTo({
    top: el.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
}

function exportMemory(messages) {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role, content: String(m.content || "") }));
}

function sessionLabel(session) {
  return session.threadId ? session.threadId.replace("session-", "会话-") : "新会话";
}

function onSelectChatFiles(event) {
  const files = Array.from(event.target?.files || []);
  pendingFiles.value = files;
}

function removePendingFile(idx) {
  pendingFiles.value.splice(idx, 1);
}

async function uploadPendingFiles(session) {
  const uploaded = [];
  for (const file of pendingFiles.value) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${apiBase}/api/files/upload`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "upload failed");
    uploaded.push({ name: data.file_name || file.name, path: data.saved_path });
  }
  if (uploaded.length > 0) {
    session.uploadedFiles.push(...uploaded);
    session.messages.push({
      role: "assistant",
      content: `已上传文件：${uploaded.map((f) => f.name).join("、")}`,
    });
  }
  pendingFiles.value = [];
  if (chatFileInput.value) chatFileInput.value.value = "";
}

async function sendMessage() {
  const text = chatInput.value.trim();
  const session = ensureCurrentSession();
  if (!text || !session || loading.value) return;

  session.messages.push({ role: "user", content: text });
  await scrollToBottom();
  session.updatedAt = Date.now();
  if (session.title === "新会话") {
    session.title = text.slice(0, 16);
  }
  chatInput.value = "";
  loading.value = true;
  let cleanupTypewriter = null;
  streamMeta.value = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };

  try {
    if (pendingFiles.value.length > 0) {
      await uploadPendingFiles(session);
    }
    const assistantMessage = { role: "assistant", content: "正在思考...\n" };
    session.messages.push(assistantMessage);
    await scrollToBottom();
    const res = await fetch(`${apiBase}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        thread_id: session.threadId || null,
        max_iterations: 2,
        uploaded_files: session.uploadedFiles.map((f) => f.path),
        memory: exportMemory(session.messages),
      }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "chat stream failed");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let latestFinal = "";
    const toolSet = new Set();
    let isSimplePath = false;
    const tokenUsage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
    let targetRenderText = "正在思考...\n";
    let revealIndex = 0;
    let renderTimer = null;
    let forcedReset = false;

    const startTypewriter = () => {
      if (renderTimer) return;
      renderTimer = setInterval(() => {
        if (forcedReset) {
          assistantMessage.content = targetRenderText;
          revealIndex = targetRenderText.length;
          forcedReset = false;
          return;
        }
        if (revealIndex >= targetRenderText.length) return;
        revealIndex = Math.min(revealIndex + 3, targetRenderText.length);
        assistantMessage.content = targetRenderText.slice(0, revealIndex);
        void scrollToBottom(false);
      }, 22);
    };

    const stopTypewriter = () => {
      if (renderTimer) {
        clearInterval(renderTimer);
        renderTimer = null;
      }
    };
    cleanupTypewriter = stopTypewriter;

    startTypewriter();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.thread_id) {
          session.threadId = evt.thread_id;
        }
        if (evt.type === "chunk" && evt.chunk && typeof evt.chunk === "object") {
          for (const [, nodeState] of Object.entries(evt.chunk)) {
            if (!nodeState || typeof nodeState !== "object") continue;
            if (nodeState.final_text) latestFinal = nodeState.final_text;
            if (typeof nodeState.is_simple_query === "boolean") {
              isSimplePath = nodeState.is_simple_query;
            }
            if (nodeState.token_usage && typeof nodeState.token_usage === "object") {
              tokenUsage.input_tokens = Number(nodeState.token_usage.input_tokens || 0);
              tokenUsage.output_tokens = Number(nodeState.token_usage.output_tokens || 0);
              tokenUsage.total_tokens = Number(nodeState.token_usage.total_tokens || 0);
            }
            for (const key of ["planner_used_tools", "extractor_used_tools", "reporter_used_tools"]) {
              const arr = nodeState[key];
              if (Array.isArray(arr)) {
                for (const name of arr) toolSet.add(String(name));
              }
            }
          }
          const toolText = toolSet.size ? `\n\n工具调用：${Array.from(toolSet).join("、")}` : "\n\n工具调用：无";
          const tokenText = `\nToken：输入 ${tokenUsage.input_tokens} / 输出 ${tokenUsage.output_tokens} / 总计 ${tokenUsage.total_tokens}`;
          streamMeta.value = {
            inputTokens: tokenUsage.input_tokens,
            outputTokens: tokenUsage.output_tokens,
            totalTokens: tokenUsage.total_tokens,
          };
          const nextTarget = (latestFinal || "正在思考...\n") + toolText + tokenText;
          if (!nextTarget.startsWith(assistantMessage.content || "")) {
            forcedReset = true;
          }
          targetRenderText = nextTarget;
        }
        if (evt.type === "error") {
          stopTypewriter();
          throw new Error(evt.detail || "stream error");
        }
      }
    }
    targetRenderText = (latestFinal || assistantMessage.content || "(空响应)").trim() || "(空响应)";
    while ((assistantMessage.content || "") !== targetRenderText) {
      await new Promise((resolve) => setTimeout(resolve, 16));
    }
    stopTypewriter();
    session.updatedAt = Date.now();
    if (!assistantMessage.content.trim()) {
      assistantMessage.content = "(空响应)";
    }
  } catch (err) {
    session.messages.push({ role: "assistant", content: `请求失败: ${err.message}` });
    await scrollToBottom();
  } finally {
    if (cleanupTypewriter) cleanupTypewriter();
    loading.value = false;
  }
}

function loadPersistedSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.sessions)) return false;
    sessions.value = parsed.sessions;
    currentSessionId.value = parsed.currentSessionId || parsed.sessions[0]?.id || "";
    return sessions.value.length > 0;
  } catch (err) {
    return false;
  }
}

watch(
  [sessions, currentSessionId],
  () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        sessions: sessions.value,
        currentSessionId: currentSessionId.value,
      }),
    );
  },
  { deep: true },
);

function onComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

async function uploadKnowledge() {
  const file = fileInput.value?.files?.[0];
  if (!file || ingesting.value) return;

  ingesting.value = true;
  const form = new FormData();
  form.append("file", file);
  form.append("tenant_id", tenantId.value);

  try {
    const res = await fetch(`${apiBase}/api/knowledge/import`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "import request failed");
    const session = ensureCurrentSession();
    session.messages.push({
      role: "assistant",
      content: `知识库导入成功: ${file.name}\nchunk数: ${data.ingest?.stored_chunks ?? "unknown"}`,
    });
    activeNav.value = "chat";
  } catch (err) {
    const session = ensureCurrentSession();
    session.messages.push({ role: "assistant", content: `导入失败: ${err.message}` });
  } finally {
    ingesting.value = false;
  }
}

if (!loadPersistedSessions()) {
  createNewSession();
}

watch(currentMessages, async () => {
  await scrollToBottom(false);
});
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="brand">DocAsst</h1>
      <button class="nav-btn" :class="{ active: activeNav === 'chat' }" @click="activeNav = 'chat'">对话</button>
      <button class="nav-btn" :class="{ active: activeNav === 'kb' }" @click="activeNav = 'kb'">导入知识库</button>

      <div class="sidebar-foot">
        <div class="session-head">
          <label>会话列表</label>
          <button @click="createNewSession">新建</button>
        </div>
        <div class="session-list">
          <button
            v-for="session in sessions"
            :key="session.id"
            class="session-item"
            :class="{ active: currentSessionId === session.id }"
            @click="selectSession(session.id)"
          >
            <div class="session-title">{{ session.title }}</div>
            <div class="session-sub">{{ sessionLabel(session) }}</div>
          </button>
        </div>
      </div>
    </aside>

    <main class="content">
      <section v-if="activeNav === 'chat'" class="chat-panel">
        <div class="stream-meta">
          <span>Token：{{ streamMeta.inputTokens }} / {{ streamMeta.outputTokens }} / {{ streamMeta.totalTokens }}</span>
        </div>
        <div ref="messagesContainer" class="messages">
          <div v-for="(item, idx) in currentMessages" :key="idx" class="msg" :class="item.role">
            {{ item.content }}
          </div>
          <div v-if="loading" class="typing-wrap">
            <div class="typing-bubble">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
        <div class="composer">
          <div class="composer-shell">
            <div v-if="pendingFiles.length" class="pending-files">
              <span v-for="(file, idx) in pendingFiles" :key="file.name + idx" class="file-chip">
                {{ file.name }}
                <button class="chip-close" @click="removePendingFile(idx)">×</button>
              </span>
            </div>
            <div class="composer-row">
              <label class="file-picker">
                <input ref="chatFileInput" type="file" multiple @change="onSelectChatFiles" />
                +
              </label>
              <textarea
                v-model="chatInput"
                rows="1"
                placeholder="给 DocAsst 发送消息（Enter 发送，Shift+Enter 换行）"
                @keydown="onComposerKeydown"
              />
              <button class="send-btn" :disabled="loading" @click="sendMessage">{{ loading ? "..." : ">" }}</button>
            </div>
          </div>
        </div>
      </section>

      <section v-else class="import-panel">
        <h2>导入知识库文件</h2>
        <p>支持 md/txt/pdf/docx/csv/json 等格式。</p>
        <label>Tenant ID</label>
        <input v-model="tenantId" />
        <input ref="fileInput" type="file" />
        <button :disabled="ingesting" @click="uploadKnowledge">{{ ingesting ? "导入中..." : "开始导入" }}</button>
      </section>
    </main>
  </div>
</template>
