(function () {
  var key = "pneumonia-app-theme";
  var saved = localStorage.getItem(key);
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  var btn = document.getElementById("themeToggle");
  if (!btn) return;
  btn.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme") || "dark";
    var next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem(key, next);
  });
})();
