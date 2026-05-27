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
const STORAGE_KEY = "docasst_web_sessions_v1";
const sessions = ref([]);
const currentSessionId = ref("");

// ── 节点标签映射 ──────────────────────────────────────────
const NODE_LABELS = {
  simpleRouter: "路由判断",
  planner: "规划",
  extractor_summarizer: "研究",
  reporter: "生成报告",
  reflection: "反思",
};
const NODE_ORDER = ["planner", "extractor_summarizer", "reporter", "reflection"];

function getNodeKeys(nodes) {
  const keys = Object.keys(nodes);
  return keys.sort((a, b) => {
    const ai = NODE_ORDER.indexOf(a);
    const bi = NODE_ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return 0;
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}

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

  try {
    if (pendingFiles.value.length > 0) {
      await uploadPendingFiles(session);
    }
    const assistantMessage = { role: "assistant", content: "", nodes: {} };
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
    const toolSet = new Set();
    let latestFinal = "";

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

        // ── 逐字 token 事件 ─────────────────────────────
        if (evt.type === "token") {
          const node = evt.node;
          if (!assistantMessage.nodes[node]) {
            assistantMessage.nodes[node] = {
              label: NODE_LABELS[node] || node,
              text: "",
              status: "streaming",
            };
          }
          assistantMessage.nodes[node].text += evt.text;
          void scrollToBottom(false);
        }

        // ── 节点完成 chunk 事件 ────────────────────────
        if (evt.type === "chunk" && evt.chunk && typeof evt.chunk === "object") {
          for (const [nodeKey, nodeState] of Object.entries(evt.chunk)) {
            if (!nodeState || typeof nodeState !== "object") continue;

            // 如果没有 token 事件回退，用 chunk 数据创建节点
            if (!assistantMessage.nodes[nodeKey]) {
              const textKey = ["final_text", "plan_text", "summary_text", "reflection_text"].find(
                (k) => nodeState[k]
              );
              if (textKey) {
                assistantMessage.nodes[nodeKey] = {
                  label: NODE_LABELS[nodeKey] || nodeKey,
                  text: nodeState[textKey],
                  status: "done",
                };
              }
            } else {
              assistantMessage.nodes[nodeKey].status = "done";
            }

            // 提取最终文本（用于 fallback content）
            if (nodeState.final_text) latestFinal = nodeState.final_text;

            // 工具调用
            for (const key of ["planner_used_tools", "extractor_used_tools", "reporter_used_tools"]) {
              const arr = nodeState[key];
              if (Array.isArray(arr)) {
                for (const name of arr) toolSet.add(String(name));
              }
            }
          }
        }

        if (evt.type === "error") {
          throw new Error(evt.detail || "stream error");
        }
      }
    }

    // 最终组装 content（兼容旧消息格式）
    const nodeTexts = Object.values(assistantMessage.nodes)
      .map((n) => n.text)
      .filter(Boolean);
    assistantMessage.content = latestFinal || nodeTexts.join("\n\n") || "(空响应)";
    session.updatedAt = Date.now();
  } catch (err) {
    session.messages.push({ role: "assistant", content: `请求失败: ${err.message}` });
    await scrollToBottom();
  } finally {
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
        <div ref="messagesContainer" class="messages">
          <div v-for="(item, idx) in currentMessages" :key="idx" class="msg" :class="item.role">
            <!-- 按节点分组显示流式输出 -->
            <template v-if="item.nodes && Object.keys(item.nodes).length">
              <div
                v-for="nodeKey in getNodeKeys(item.nodes)"
                :key="nodeKey"
                class="node-section"
                :class="item.nodes[nodeKey].status"
              >
                <div class="node-label">
                  {{ item.nodes[nodeKey].label }}
                  <span v-if="item.nodes[nodeKey].status === 'streaming'" class="node-indicator"></span>
                </div>
                <div class="node-text">{{ item.nodes[nodeKey].text }}</div>
              </div>
            </template>
            <!-- 兼容无 nodes 的旧消息 -->
            <template v-else>
              {{ item.content }}
            </template>
          </div>
          <div v-if="loading && !currentMessages.some(m => m.nodes && Object.keys(m.nodes).length)" class="typing-wrap">
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
