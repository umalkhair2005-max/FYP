/**
 * Patient records table — ⋮ row actions (fixed-position menu, avoids table overflow clip).
 */
(function () {
  function closeAll() {
    document.querySelectorAll(".pr-row-menu.is-open").forEach(function (w) {
      w.classList.remove("is-open");
      var b = w.querySelector(".pr-menu-trigger");
      if (b) b.setAttribute("aria-expanded", "false");
      var m = w.querySelector(".pr-menu-dropdown");
      if (m) {
        m.style.top = "";
        m.style.left = "";
        m.style.right = "";
        m.style.position = "";
        m.style.zIndex = "";
        m.style.visibility = "";
      }
    });
  }

  function placeMenu(wrap) {
    var btn = wrap.querySelector(".pr-menu-trigger");
    var menu = wrap.querySelector(".pr-menu-dropdown");
    if (!btn || !menu) return;
    menu.style.visibility = "hidden";
    menu.style.position = "fixed";
    menu.style.zIndex = "13000";
    requestAnimationFrame(function () {
      var br = btn.getBoundingClientRect();
      var mw = menu.offsetWidth || 160;
      var mh = menu.offsetHeight || 88;
      var left = br.right - mw;
      if (left < 8) left = 8;
      if (left + mw > window.innerWidth - 8) left = window.innerWidth - mw - 8;
      var top = br.bottom + 6;
      if (top + mh > window.innerHeight - 8) top = Math.max(8, br.top - mh - 6);
      menu.style.top = top + "px";
      menu.style.left = left + "px";
      menu.style.visibility = "";
    });
  }

  document.addEventListener(
    "click",
    function (e) {
      var trigger = e.target.closest(".pr-menu-trigger");
      if (trigger) {
        e.preventDefault();
        e.stopPropagation();
        var wrap = trigger.closest(".pr-row-menu");
        var wasOpen = wrap.classList.contains("is-open");
        closeAll();
        if (!wasOpen) {
          wrap.classList.add("is-open");
          trigger.setAttribute("aria-expanded", "true");
          placeMenu(wrap);
        }
        return;
      }

      if (e.target.closest(".pr-menu-item")) {
        closeAll();
        return;
      }

      if (e.target.closest(".pr-row-menu")) return;
      closeAll();
    },
    false
  );

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeAll();
  });

  window.addEventListener(
    "scroll",
    function () {
      closeAll();
    },
    true
  );
  window.addEventListener("resize", closeAll);
})();
