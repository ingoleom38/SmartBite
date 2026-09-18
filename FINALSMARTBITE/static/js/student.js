// student.js — tiny progressive-enhancement helper for the collapsible
// mobile sidebar. The dashboard works without JS; this only improves
// the mobile nav toggle.
(function () {
  var sidebar = document.getElementById("sidebar");
  var hamburger = document.getElementById("hamburger");
  var closeBtn = document.getElementById("sidebarClose");
  if (!sidebar) return;
  function open() { sidebar.classList.add("open"); }
  function close() { sidebar.classList.remove("open"); }
  if (hamburger) hamburger.addEventListener("click", open);
  if (closeBtn) closeBtn.addEventListener("click", close);
  document.addEventListener("click", function (e) {
    if (sidebar.classList.contains("open") && !sidebar.contains(e.target) && e.target !== hamburger) {
      close();
    }
  });
})();
