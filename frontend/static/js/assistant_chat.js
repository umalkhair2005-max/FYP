(function () {
  var input = document.getElementById("assistantInput");
  var send = document.getElementById("assistantSend");
  var clearBtn = document.getElementById("assistantClear");
  var messages = document.getElementById("assistantMessages");
  var suggestionsEl = document.getElementById("assistantSuggestions");
  var micBtn = document.getElementById("assistantMic");
  var speakBtn = document.getElementById("assistantSpeakLast");
  var langSelect = document.getElementById("assistantVoiceLang");
  var autoSpeakCheckbox = document.getElementById("assistantAutoSpeak");
  var AUTO_SPEAK_LS = "pneumonia-assistant-auto-speak";
  if (!input || !send || !messages || !suggestionsEl) return;

  var Recognition =
    typeof window !== "undefined"
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null;
  var recognition = null;
  var listening = false;
  var voiceBuffer = "";

  function getVoiceLang() {
    if (langSelect && langSelect.value && langSelect.value !== "auto") {
      return langSelect.value;
    }
    return (typeof navigator !== "undefined" && navigator.language) || "en-US";
  }

  function setChatBusy(busy) {
    send.disabled = busy;
    if (micBtn) {
      micBtn.disabled = busy || !Recognition;
      if (busy && listening && recognition) {
        try {
          recognition.stop();
        } catch (e) {}
      }
    }
    if (speakBtn) speakBtn.disabled = busy || !window.speechSynthesis;
  }

  if (Recognition && micBtn) {
    recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    function syncRecoLang() {
      recognition.lang = getVoiceLang();
    }

    recognition.onstart = function () {
      listening = true;
      micBtn.classList.add("is-listening");
      micBtn.setAttribute("aria-pressed", "true");
      micBtn.setAttribute("title", "Rokne ke liye dubara dabayein");
    };

    recognition.onend = function () {
      listening = false;
      micBtn.classList.remove("is-listening");
      micBtn.setAttribute("aria-pressed", "false");
      micBtn.setAttribute("title", "Bol kar likhein (voice)");
    };

    recognition.onerror = function (ev) {
      listening = false;
      micBtn.classList.remove("is-listening");
      micBtn.setAttribute("aria-pressed", "false");
      if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
        appendBubble(
          "Microphone block hai. Browser / site settings mein allow karke phir mic try karein.",
          "bot",
        );
      }
    };

    recognition.onresult = function (ev) {
      for (var i = ev.resultIndex; i < ev.results.length; i++) {
        if (!ev.results[i].isFinal) continue;
        var t = (ev.results[i][0] && ev.results[i][0].transcript) || "";
        t = t.trim();
        if (!t) continue;
        voiceBuffer = voiceBuffer.trim() ? voiceBuffer.trim() + " " + t : t;
        input.value = voiceBuffer;
      }
    };

    micBtn.addEventListener("click", function () {
      if (!recognition) return;
      if (listening) {
        try {
          recognition.stop();
        } catch (e) {}
        return;
      }
      try {
        if (window.speechSynthesis) window.speechSynthesis.cancel();
      } catch (e) {}
      voiceBuffer = input.value || "";
      syncRecoLang();
      try {
        recognition.start();
      } catch (e) {
        appendBubble("Voice start nahi ho saka. Thori dair baad mic dubara dabayein.", "bot");
      }
    });

    if (langSelect) {
      langSelect.addEventListener("change", function () {
        if (listening) {
          try {
            recognition.stop();
          } catch (e) {}
        }
      });
    }
  } else if (micBtn) {
    micBtn.disabled = true;
    micBtn.title = "Voice typing is liye Chrome ya Edge use karein.";
  }

  var welcomeText =
    "Assistant ready (internet + API key). Mic = bol kar type. Upar ✓ Jawab voice mein sunen = har naya jawab bol kar sunega. 🔊 = sirf aakhri jawab dubara sunen.";

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

  function cancelSpeech() {
    try {
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    } catch (e) {}
  }

  /** force=true for manual speaker button (ignores checkbox). */
  function speakAssistReply(raw, force) {
    if (!window.speechSynthesis) return;
    var text = (raw || "").trim();
    if (!text) return;
    if (!force && (!autoSpeakCheckbox || !autoSpeakCheckbox.checked)) return;
    cancelSpeech();
    try {
      var u = new SpeechSynthesisUtterance(text);
      u.lang = getVoiceLang();
      u.rate = 0.93;
      window.speechSynthesis.speak(u);
    } catch (e) {}
  }

  if (speakBtn) {
    if (!window.speechSynthesis) {
      speakBtn.disabled = true;
      speakBtn.title = "Is browser mein read-aloud maujood nahi.";
    } else {
      speakBtn.addEventListener("click", function () {
        var nodes = messages.querySelectorAll(".chat-bubble.bot");
        if (!nodes.length) return;
        var last = nodes[nodes.length - 1];
        var text = (last.textContent || "").trim();
        speakAssistReply(text, true);
      });
    }
  }

  if (autoSpeakCheckbox && typeof localStorage !== "undefined") {
    var preset = localStorage.getItem(AUTO_SPEAK_LS);
    if (preset === "1") autoSpeakCheckbox.checked = true;
    if (preset === "0") autoSpeakCheckbox.checked = false;
    autoSpeakCheckbox.addEventListener("change", function () {
      localStorage.setItem(AUTO_SPEAK_LS, autoSpeakCheckbox.checked ? "1" : "0");
    });
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
    setChatBusy(true);
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
          speakAssistReply(text, false);
          return;
        }
        appendBubble(text, "bot");
        speakAssistReply(text, false);
        if (data.suggestions && data.suggestions.length) renderChips(data.suggestions);
      })
      .catch(function () {
        appendBubble("Could not reach the server. Check your connection and try again.", "bot");
      })
      .finally(function () {
        setChatBusy(false);
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
