(function () {
  var key = "pneumonia-app-theme";
  function syncThemeColor() {
    var t = document.documentElement.getAttribute("data-theme") || "dark";
    var m = document.querySelector('meta[name="theme-color"]');
    if (m) {
      m.setAttribute("content", t === "light" ? "#f8fafc" : "#0c1222");
    }
  }
  window.pneumoniaSyncThemeColor = syncThemeColor;
  var saved = localStorage.getItem(key);
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  syncThemeColor();
  var btn = document.getElementById("themeToggle");
  if (btn) {
    btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme") || "dark";
      var next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem(key, next);
      syncThemeColor();
    });
  }
})();
