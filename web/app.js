"use strict";

const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
const clearButton = document.getElementById("clear");
const temperature = document.getElementById("temp");
const temperatureValue = document.getElementById("tempv");
const showReasoning = document.getElementById("thinking");
const statusLabel = document.getElementById("status");
const subtitle = document.getElementById("subtitle");
const title = document.getElementById("title");

const THINK_END = "</think>";

// Only final answers go in here, never reasoning, so the model is not fed its
// own scratchpad on the next turn.
let history = [];
let controller = null;

// Replaced by whatever /health reports, so this page works against any model.
let modelId = "";

temperature.oninput = () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
};

function setStatus(text, kind = "") {
  statusLabel.textContent = text;
  statusLabel.className = "status " + kind;
}

function addBubble(role, text = "") {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function addReasoningBox() {
  const div = document.createElement("div");
  div.className = "reasoning";
  chat.appendChild(div);
  return div;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

/*
 * Split raw Qwen output into reasoning and answer.
 *
 * The server streams the model's raw text, so the reasoning section arrives
 * first and is closed by `</think>`. Until that tag shows up we cannot tell
 * reasoning from answer, so everything is provisionally reasoning; the caller
 * reruns this on the full buffer once the stream ends.
 */
function formatToolCalls(calls) {
  return calls
    .filter((call) => call?.function?.name)
    .map((call) => {
      let args = call.function.arguments || "";
      try {
        args = JSON.stringify(JSON.parse(args), null, 2);
      } catch (_) {}
      return "⚙ " + call.function.name + (args ? "\n" + args : "");
    })
    .join("\n\n");
}

function mergeToolCallDelta(toolCalls, parts) {
  for (const part of parts) {
    const index = part.index ?? toolCalls.length;
    if (!toolCalls[index]) {
      toolCalls[index] = {
        id: part.id,
        type: part.type || "function",
        function: { name: "", arguments: "" },
      };
    }
    const slot = toolCalls[index];
    if (part.id) {
      slot.id = part.id;
    }
    if (part.function?.name) {
      slot.function.name += part.function.name;
    }
    if (part.function?.arguments) {
      slot.function.arguments += part.function.arguments;
    }
  }
}

function renderAssistant(answerBox, reasoningBox, raw, toolCalls) {
  const split = splitReasoning(raw);
  if (showReasoning.checked) {
    reasoningBox.textContent = split.reasoning;
    reasoningBox.style.display = split.reasoning ? "block" : "none";
  }
  const answer = raw.includes(THINK_END) ? split.answer : raw;
  const tools = formatToolCalls(toolCalls);
  answerBox.textContent = [answer.trim(), tools].filter(Boolean).join("\n\n");
  scrollToBottom();
}

function splitReasoning(text) {
  const end = text.indexOf(THINK_END);

  if (end === -1) {
    return { reasoning: text.replace("<think>", "").trim(), answer: "" };
  }

  return {
    reasoning: text.slice(0, end).replace("<think>", "").trim(),
    answer: text.slice(end + THINK_END.length).trimStart(),
  };
}

function clearChat() {
  if (controller) {
    controller.abort();
    controller = null;
  }
  history = [];
  chat.innerHTML = "";
  sendButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Ready");
  input.focus();
}

clearButton.onclick = clearChat;

stopButton.onclick = () => {
  if (controller) {
    controller.abort();
    controller = null;
  }
};

async function sendMessage() {
  const message = input.value.trim();
  if (!message || sendButton.disabled) {
    return;
  }

  input.value = "";
  addBubble("user", message);

  const reasoningBox = addReasoningBox();
  const answerBox = addBubble("assistant", "");

  if (showReasoning.checked) {
    reasoningBox.style.display = "block";
  }

  sendButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Generating...");

  history.push({ role: "user", content: message });
  controller = new AbortController();

  let raw = "";
  let toolCalls = [];
  const started = performance.now();

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: modelId,
        messages: history,
        temperature: Number(temperature.value),
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Server-sent events are separated by a blank line. A trailing partial
      // event stays in the buffer until the rest of it arrives.
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const event of events) {
        for (const line of event.split("\n")) {
          if (!line.startsWith("data: ")) {
            continue;
          }

          const payload = line.slice(6);
          if (payload === "[DONE]") {
            continue;
          }

          let parsed;
          try {
            parsed = JSON.parse(payload);
          } catch (_) {
            continue;
          }

          if (parsed.error) {
            throw new Error(parsed.error.message || "server error");
          }

          const delta = parsed.choices?.[0]?.delta || {};
          if (delta.tool_calls) {
            mergeToolCallDelta(toolCalls, delta.tool_calls);
          }
          const piece = delta.content || "";
          if (piece) {
            raw += piece;
          }
          if (!piece && !delta.tool_calls) {
            continue;
          }

          renderAssistant(answerBox, reasoningBox, raw, toolCalls);
        }
      }
    }

    renderAssistant(answerBox, reasoningBox, raw, toolCalls);
    const visible = answerBox.textContent.trim();
    history.push({ role: "assistant", content: visible });

    const seconds = (performance.now() - started) / 1000;
    setStatus(`Done in ${seconds.toFixed(1)}s`, "ok");
  } catch (error) {
    history.pop();

    if (error.name === "AbortError") {
      setStatus("Stopped");
      return;
    }

    answerBox.textContent = "Error: " + error.message;
    setStatus("Error", "error");
  } finally {
    controller = null;
    sendButton.disabled = false;
    stopButton.disabled = true;
    input.focus();
  }
}

sendButton.onclick = sendMessage;

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

fetch("/health")
  .then((response) => response.json())
  .then((info) => {
    modelId = info.model || modelId;
    document.title = info.model || "gguf-serve";
    title.textContent = info.model || "gguf-serve";
    subtitle.textContent = `${info.model_file} · ${info.context} token context · OpenAI-compatible API`;
  })
  .catch(() => {});

input.focus();
