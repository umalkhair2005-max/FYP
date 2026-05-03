/**
 * Themed confirm dialog (replaces native confirm()).
 * window.showHcConfirm({ title, message, confirmLabel, cancelLabel, variant })
 *   → Promise<boolean>
 */
(function () {
  var prevFocus = null;

  function ensureModal() {
    var root = document.getElementById("hcConfirmRoot");
    if (root) return root;

    document.body.insertAdjacentHTML(
      "beforeend",
        '<div id="hcConfirmRoot" class="hc-confirm" aria-hidden="true">' +
        '<div class="hc-confirm-backdrop"></div>' +
        '<div class="hc-confirm-panel" role="dialog" aria-modal="true" aria-labelledby="hcConfirmTitleEl">' +
        '<div class="hc-confirm-icon-wrap"><span class="hc-confirm-icon" aria-hidden="true">⬡</span></div>' +
        '<h2 id="hcConfirmTitleEl" class="hc-confirm-title">Confirm</h2>' +
        '<p id="hcConfirmBodyEl" class="hc-confirm-body"></p>' +
        '<div class="hc-confirm-actions">' +
        '<button type="button" class="hc-confirm-btn hc-confirm-btn-cancel">Cancel</button>' +
        '<button type="button" class="hc-confirm-btn hc-confirm-btn-ok">Continue</button>' +
        "</div></div></div>"
    );
    return document.getElementById("hcConfirmRoot");
  }

  function closeModal(root, resolve, value) {
    root.classList.remove("is-open", "hc-confirm--danger");
    root.setAttribute("aria-hidden", "true");
    document.body.classList.remove("hc-confirm-open");
    if (prevFocus && prevFocus.focus) {
      try {
        prevFocus.focus();
      } catch (e1) {}
    }
    prevFocus = null;
    if (resolve) resolve(value);
  }

  window.showHcConfirm = function (opts) {
    opts = opts || {};
    var title = opts.title || "Confirm";
    var message = opts.message || "";
    var confirmLabel = opts.confirmLabel || "Continue";
    var cancelLabel = opts.cancelLabel || "Cancel";
    var variant = opts.variant === "danger" ? "danger" : "default";

    return new Promise(function (resolve) {
      var root = ensureModal();
      var titleEl = document.getElementById("hcConfirmTitleEl");
      var bodyEl = document.getElementById("hcConfirmBodyEl");
      var btnOk = root.querySelector(".hc-confirm-btn-ok");
      var btnCancel = root.querySelector(".hc-confirm-btn-cancel");
      var backdrop = root.querySelector(".hc-confirm-backdrop");
      if (!btnOk || !btnCancel || !backdrop) {
        resolve(false);
        return;
      }

      if (titleEl) titleEl.textContent = title;
      if (bodyEl) bodyEl.textContent = message;
      if (btnOk) btnOk.textContent = confirmLabel;
      if (btnCancel) btnCancel.textContent = cancelLabel;

      root.classList.toggle("hc-confirm--danger", variant === "danger");
      root.setAttribute("aria-hidden", "false");
      document.body.classList.add("hc-confirm-open");

      prevFocus = document.activeElement;
      requestAnimationFrame(function () {
        root.classList.add("is-open");
        if (btnOk) btnOk.focus();
      });

      function onOk() {
        cleanup();
        closeModal(root, resolve, true);
      }
      function onCancel() {
        cleanup();
        closeModal(root, resolve, false);
      }

      function onKey(e) {
        if (e.key === "Escape") {
          e.preventDefault();
          onCancel();
        }
      }

      function cleanup() {
        btnOk.removeEventListener("click", onOk);
        btnCancel.removeEventListener("click", onCancel);
        backdrop.removeEventListener("click", onCancel);
        document.removeEventListener("keydown", onKey);
      }

      btnOk.addEventListener("click", onOk);
      btnCancel.addEventListener("click", onCancel);
      backdrop.addEventListener("click", onCancel);
      document.addEventListener("keydown", onKey);
    });
  };

  document.addEventListener(
    "submit",
    function (e) {
      var form = e.target;
      if (!form || form.nodeName !== "FORM") return;
      var msg = form.getAttribute("data-hc-confirm");
      if (!msg) return;
      e.preventDefault();
      e.stopPropagation();

      var title = form.getAttribute("data-hc-confirm-title") || "Confirm action";
      var danger = form.hasAttribute("data-hc-danger");
      var confirmLabel = form.getAttribute("data-hc-confirm-ok") || (danger ? "Delete" : "Continue");

      window
        .showHcConfirm({
          title: title,
          message: msg,
          variant: danger ? "danger" : "default",
          confirmLabel: confirmLabel,
          cancelLabel: "Cancel",
        })
        .then(function (ok) {
          if (ok) form.submit();
        });
    },
    true
  );
})();
