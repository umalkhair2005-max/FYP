(function () {
  var input = document.getElementById("assistantInput");
  var send = document.getElementById("assistantSend");
  var clearBtn = document.getElementById("assistantClear");
  var messages = document.getElementById("assistantMessages");
  var suggestionsEl = document.getElementById("assistantSuggestions");
  if (!input || !send || !messages || !suggestionsEl) return;

  var welcomeText =
    "Assistant ready (internet + API key). You can type in English or Roman Urdu (Latin letters). Replies will match your language. Use chips below.";

  var defaultChips = [
    "What is today's date?",
    "Aaj ki date kya hai?",
    "Aaj konsa din hai?",
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

  /** Same intent rules as backend _is_clock_related_question (must stay in sync). */
  function matchClockIntent(raw) {
    var m = raw.trim().toLowerCase();
    if (!m) return null;

    var asksDate =
      /\b(today'?s date|todays date|what is today'?s date|date today|what'?s the date|whats the date|what date(\s+is it)?|current date)\b/.test(
        m,
      ) || /\b(?:aaj|aj)\s+ki\s+(?:date|tareekh|tarikh)\b/.test(m);

    var asksDay =
      /\b(what day(\s+is it)?|which day(\s+is it)?|day is it|day today)\b/.test(m) ||
      /\b(?:aaj|aj)\s+konsa\s+din\b/.test(m) ||
      /\b(?:aaj|aj)\s+kaun\s+sa\s+din\b/.test(m) ||
      /\b(?:aaj|aj)\s+din\s+(?:kya|hai|hay)\b/.test(m);

    var asksTime =
      /\b(current time|what'?s the time|what time(\s+is it)?|time now|waqt|kitne baje|abhi kya time)\b/.test(m);

    if (!asksDate && !asksDay && !asksTime) return null;
    return { asksDate: asksDate, asksDay: asksDay, asksTime: asksTime };
  }

  function isClockRelatedQuestion(text) {
    return matchClockIntent(text) !== null;
  }

  function isRomanUrduClockStyle(raw) {
    var m = raw.trim().toLowerCase();
    return /\b(aaj|aj|tareekh|tarikh|waqt|kya|hai|hay|konsa|kab|din|kaun|baje)\b/.test(m);
  }

  /**
   * Browser clock via new Date() — never hardcoded. en-US formatting for calendar parts (example: May 9, 2026).
   */
  function buildClientClockReply(originalText) {
    var intent = matchClockIntent(originalText);
    if (!intent) return null;

    var now = new Date();
    var dateEnUs = now.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    var weekdayEn = now.toLocaleDateString("en-US", { weekday: "long" });
    var timeEnUs = now.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
    });

    var roman = isRomanUrduClockStyle(originalText);
    var parts = [];

    if (roman) {
      if (intent.asksDate || intent.asksDay) {
        parts.push("Aaj ki date " + dateEnUs + " hai aur aaj " + weekdayEn + " hai.");
      }
      if (intent.asksTime) {
        if (intent.asksDate || intent.asksDay) {
          parts.push("Waqt abhi " + timeEnUs + " hai.");
        } else {
          parts.push("Abhi waqt " + timeEnUs + " hai.");
        }
      }
      return parts.join(" ");
    }

    if (intent.asksDate || intent.asksDay) {
      parts.push("Today's date is " + dateEnUs + ", and today is " + weekdayEn + ".");
    }
    if (intent.asksTime) {
      parts.push("Your browser's local time is " + timeEnUs + ".");
    }
    return parts.join(" ");
  }

  function sendMessage() {
    var t = input.value.trim();
    if (!t) return;
    appendBubble(t, "user");
    input.value = "";
    send.disabled = true;
    var payload = { message: t };
    if (isClockRelatedQuestion(t)) {
      var clockAnswer = buildClientClockReply(t);
      if (clockAnswer) payload.client_clock_answer = clockAnswer;
    }
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
