/**
 * Healthcare shell: profile menu, FAB, stat counters (optional clock if #hcClock exists).
 */
(function () {
  function pad(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function tickClock() {
    var el = document.getElementById("hcClock");
    if (!el) return;
    var d = new Date();
    var days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    var months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    el.textContent =
      days[d.getDay()] +
      " " +
      months[d.getMonth()] +
      " " +
      d.getDate() +
      ", " +
      pad(d.getHours()) +
      ":" +
      pad(d.getMinutes()) +
      ":" +
      pad(d.getSeconds());
  }
  tickClock();
  setInterval(tickClock, 1000);

  var profBtn = document.getElementById("hcProfileBtn");
  var profWrap = document.getElementById("hcProfile");
  if (profBtn && profWrap) {
    profBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      profWrap.classList.toggle("is-open");
    });
    document.addEventListener("click", function () {
      profWrap.classList.remove("is-open");
    });
  }

  var fabMain = document.getElementById("hcFabMain");
  var fabPanel = document.getElementById("hcFabPanel");
  if (fabMain && fabPanel) {
    fabMain.addEventListener("click", function (e) {
      e.stopPropagation();
      fabPanel.classList.toggle("is-open");
    });
    document.addEventListener("click", function () {
      fabPanel.classList.remove("is-open");
    });
  }

  function animateValue(el, end, duration) {
    if (!el) return;
    var start = 0;
    var range = end - start;
    var t0 = null;
    function step(ts) {
      if (!t0) t0 = ts;
      var p = Math.min(1, (ts - t0) / duration);
      var ease = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(start + range * ease);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll("[data-count-to]").forEach(function (el) {
    var v = parseInt(el.getAttribute("data-count-to"), 10);
    if (isNaN(v)) return;
    el.textContent = "0";
    setTimeout(function () {
      animateValue(el, v, 900);
    }, 200);
  });
})();
