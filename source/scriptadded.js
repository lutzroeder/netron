function addbutton() {
  if (document.getElementById("graph")) {
    if (document.getElementById("origin")) {
      if (document.getElementById("nodes")) {
        var child = document.getElementById("nodes").children[0];
        if (document.getElementById("input-name-input").children[0].getAttribute("listener") !== 'true') {
          document.getElementById("input-name-input").children[0].setAttribute('listener', 'true');
          document.getElementById("input-name-input").children[0].addEventListener("click", function() {
            console.log("am apasat pe input");
            var sidebarobj = document.getElementById("sidebar-content").children[0];
            var flag = 0;
            for (var i = 0; i < sidebarobj.children.length; i++) {
              if (sidebarobj.children[i].type == "button") {
                flag = 1;
                if (sidebarobj.children[i].getAttribute('listener') !== 'true') {
                  sidebarobj.children[i].setAttribute('listener', 'true');
                  sidebarobj.children[i].addEventListener('click', function() {
                    // var value = JSON.parse(h.target.value);
                    console.log("am apasat pe buton");
                  });
                }
              }
            }
            if (flag == 0) {
              var newchild = document.createElement('button');
              newchild.type = "button";
              //newchild.textContent = stringnew["button_name"];
              newchild.style = "position: relative;";
              sidebarobj.appendChild(newchild);
            }
          });
        }
      }
    }
  }
}
window.setInterval(addbutton, 100);