(function () {
  var input = document.getElementById("assistantInput");
  var send = document.getElementById("assistantSend");
  var clearBtn = document.getElementById("assistantClear");
  var messages = document.getElementById("assistantMessages");
  var suggestionsEl = document.getElementById("assistantSuggestions");
  if (!input || !send || !messages || !suggestionsEl) return;

  var welcomeText =
    "Online assistant is ready. Ask in English. Use the chips below. If a reply fails, check internet and your API key.";

  var defaultChips = [
    "What is today's date?",
    "What should a patient with pneumonia do in general?",
    "What are common pneumonia symptoms?",
  ];

  function appendBubble(text, who) {
    var d = document.createElement("div");
    d.className = "chat-bubble " + (who === "user" ? "user" : "bot");
    d.textContent = text;
    messages.appendChild(d);
    messages.scrollTop = messages.scrollHeight;
  }

  function renderChips(labels) {
    suggestionsEl.innerHTML = "";
    var list = labels && labels.length ? labels : defaultChips;
    list.slice(0, 4).forEach(function (label) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "assistant-chip";
      b.textContent = label;
      b.addEventListener("click", function () {
        input.value = label;
        sendMessage();
      });
      suggestionsEl.appendChild(b);
    });
  }

  function renderHistory(list) {
    messages.innerHTML = "";
    if (!list || list.length === 0) {
      appendBubble(welcomeText, "bot");
      renderChips(defaultChips);
      return;
    }
    list.forEach(function (m) {
      var who = m.role === "user" ? "user" : "bot";
      appendBubble(m.content || "", who);
    });
    renderChips(defaultChips);
  }

  function loadHistory() {
    fetch("/api/chat/history")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        renderHistory(data.messages);
      })
      .catch(function () {
        renderHistory([]);
      });
  }

  function sendMessage() {
    var t = input.value.trim();
    if (!t) return;
    appendBubble(t, "user");
    input.value = "";
    send.disabled = true;
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: t }),
    })
      .then(function (r) {
        return r.json().then(function (data) {
          return { ok: r.ok, data: data };
        });
      })
      .then(function (res) {
        var data = res.data;
        var text = data.reply || data.error || "(No reply.)";
        if (!res.ok) {
          appendBubble(text, "bot");
          return;
        }
        appendBubble(text, "bot");
        if (data.suggestions && data.suggestions.length) renderChips(data.suggestions);
      })
      .catch(function () {
        appendBubble("Could not reach the server. Check your connection and try again.", "bot");
      })
      .finally(function () {
        send.disabled = false;
        input.focus();
      });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      function runClear() {
        fetch("/api/chat/history", { method: "DELETE" })
          .then(function () {
            renderHistory([]);
          })
          .catch(function () {
            appendBubble("Could not clear history.", "bot");
          });
      }
      if (typeof window.showHcConfirm === "function") {
        window
          .showHcConfirm({
            title: "Clear conversation",
            message: "Clear this conversation? This cannot be undone.",
            variant: "danger",
            confirmLabel: "Clear",
            cancelLabel: "Cancel",
          })
          .then(function (ok) {
            if (ok) runClear();
          });
      } else if (window.confirm("Clear this conversation? This cannot be undone.")) {
        runClear();
      }
    });
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") sendMessage();
  });

  loadHistory();
})();
