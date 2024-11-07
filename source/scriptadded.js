class Scr {
  constructor() {}

  static addbutton() {
    var listofindexes = [];
    if (document.getElementById("graph")) {
      if (document.getElementById("origin")) {
        if (document.getElementById("nodes")) {
          if (listofindexes.length == 0) {
            for (var i = 1; i < document.getElementById("nodes").children.length; i++) {
              listofindexes.push(document.getElementById("nodes").children[i].id);
            }
          }
          var child = document.getElementById("nodes").children[0];
          var path;
          var flag = 0;
          if (child.children[0].getAttribute("listener") !== 'true') {
            child.children[0].setAttribute('listener', 'true');
            child.children[0].addEventListener("click", function() {
              console.log("am apasat pe input");
              var sidebarobj = document.getElementById("sidebar-content").children[0];
              for (var i = 0; i < sidebarobj.children.length; i++) {
                if (sidebarobj.children[i].type == "button") {
                  flag = 1;
                }
              }
              if (flag == 0) {
                var newchild = document.createElement('button');
                newchild.type = "button";
                newchild.innerText = "Button for python script";
                newchild.webkitdirectory = 'true';
                if (newchild.getAttribute('listener') !== 'true') {
                  newchild.setAttribute('listener', 'true');
                  newchild.addEventListener('click', function() {
                      path = prompt("Please provide the path"); //path to 'results' file
                      var listnodes = document.getElementById("nodes").children;
                      for (var i = 1; i < listnodes.length; i++) {
                        listnodes[i].dataset.path = path;
                        listnodes[i].dataset.newpath = "C:\Users\nxg06533\OneDrive - NXP\Desktop\results\mobilenet_v1_1.0_224_int8_imx95_conv\tensor_indexes";
                        listnodes[i].dataset.pathtopictures = "C:\Users\nxg06533\OneDrive - NXP\Desktop\results\mobilenet_v1_1.0_224_int8_imx95_compare_tensors_results\ILSVRC2012_val_00000001";
                      }
                  });
                }
                sidebarobj.appendChild(newchild);
              }
            });
          }
          if (document.getElementById("list-attributes").children) {
            if (document.getElementById("sidebar-content")) {
              if (document.getElementById("sidebar-content").children[0]) {
                if (document.getElementById("sidebar-content").children[0].children) {
                  var sidebarobj = document.getElementById("sidebar-content").children[0];
                    if (typeof document.getElementsByClassName("node graph-node select") !== "undefined" && document.getElementsByClassName("node graph-node select").length !== 0) {
                      console.log(document.getElementsByClassName("node graph-node select"));
                      if (document.getElementsByClassName("node graph-node select")[0] !== document.getElementById("graph").children[0]) {
                        if (document.getElementsByClassName("node graph-node select")[0].hasAttribute('data-path')) {
                          var variable = document.getElementsByClassName("node graph-node select")[0].id;
                          console.log(variable);
                          var k = 0;
                          for (var i = 0; i < sidebarobj.children.length; i++) {
                            if (sidebarobj.children[i].name == 'heatmaps_button') {
                              k = 1;
                            }
                            }
                            if (k == 0) {
                              const childmeta = document.createElement('button');
                              childmeta.name = 'heatmaps_button';
                              childmeta.innerText = "Open histogram and heatmap";
                              childmeta.addEventListener("click", function() {
                                console.log("am apasat pe operator");
                              })
                              sidebarobj.appendChild(childmeta);
                            }
                          }
                          
                        }
                      }
                    
                }
              }
            }
          }
        }
      }
    }
  }

  static welp() {
    window.setInterval(Scr.addbutton, 100);
  }
  
}

Scr.welp();