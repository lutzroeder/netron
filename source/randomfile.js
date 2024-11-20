function image() {
  if (document.getElementById("graph")) {
    if (document.getElementById("origin")) {
      if (document.getElementById("nodes") && document.getElementById("edge-paths")) {
        if (document.getElementById("list-attributes").innerHTML.length !== 0) {
          var parent_n = document.getElementById("nodes");
          var parent_t = document.getElementById("edge-paths");
          var lista = document.getElementById("list-attributes");
          var onmouseoverdict = {};
          for (var i = 0; i < lista.children.length; i++) {
            var child = lista.children[i];
            var inner = JSON.parse(child.innerHTML);
            var item;
            var elem;
            var keys = Object.keys(inner);
            for (var j = 0; j < keys.length; j++) {
              if (keys[j].startsWith("img_")) {
                var getting = JSON.parse(inner[keys[j]]);
                if (getting["class"] == "operator") {
                  var value = "node-id-" + getting["id"];
                  if (!(value in onmouseoverdict)) {
                    onmouseoverdict[value] = [];
                  }
                  onmouseoverdict[value].push("image-node-id-" + getting["id"] + "_" + keys[j]);
                } else {
                  var value = getting["id"] + 1;
                  if (!(value in onmouseoverdict)) {
                    onmouseoverdict[value] = [];
                  }
                  onmouseoverdict[value].push("tensor-image-" + getting["id"] + "_" + keys[j]);
                }
              }
            }
          }
          for (const [elem, lista] of Object.entries(onmouseoverdict)) {
            console.log(elem);
            console.log(lista);
            document.getElementById(elem).onmouseover = function() {
              for (const item of lista) {
                document.getElementById(item).style.display = "block";
              }
            }
            document.getElementById(elem).onmouseout = function() {
              for (const item of lista) {
                document.getElementById(item).style.display = "none";
              }
            }
          }
        }
      }
    }
  }
}

window.setInterval(image, 1);