(function () {
  var fab = document.getElementById("chatFab");
  var panel = document.getElementById("chatPanel");
  var input = document.getElementById("chatInput");
  var send = document.getElementById("chatSend");
  var messages = document.getElementById("chatMessages");
  if (!fab || !panel || !input || !send || !messages) return;

  function appendBubble(text, who) {
    var d = document.createElement("div");
    d.className = "chat-bubble " + (who === "user" ? "user" : "bot");
    d.textContent = text;
    messages.appendChild(d);
    messages.scrollTop = messages.scrollHeight;
  }

  function togglePanel() {
    var open = panel.classList.toggle("is-open");
    panel.setAttribute("aria-hidden", open ? "false" : "true");
  }

  fab.addEventListener("click", togglePanel);

  var side = document.getElementById("sidebarChatLink");
  if (side) {
    side.addEventListener("click", function (e) {
      e.preventDefault();
      if (!panel.classList.contains("is-open")) togglePanel();
      input.focus();
    });
  }

  function sendMessage() {
    var t = input.value.trim();
    if (!t) return;
    appendBubble(t, "user");
    input.value = "";
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: t }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        appendBubble(data.reply || "(No reply.)", "bot");
      })
      .catch(function () {
        appendBubble("Could not reach assistant. Try again.", "bot");
      });
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") sendMessage();
  });
})();
