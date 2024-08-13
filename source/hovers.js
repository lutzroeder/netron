class SVGPathElement extends HTMLElement {}

window.SVGPathElement = SVGPathElement

function functie() {
  let input = document.createElement('input');
  input.type = 'file';
  input.onchange = _ => {
    let files =   Array.from(input.files);
    const reader = new FileReader();
    reader.onload = function() {
      const content = reader.result;
      var lines = content.split('\n');
      var id, meta, style, specifier;
      var childmeta;
      var nofb = 0;
      var otherstring = JSON.stringify({});
      for (var line = 0; line < lines.length; line++) {
        var lineread = lines[line].split(":");
        var first = lineread[0] == undefined? '' : lineread[0].trim();
        
        if (first.toLowerCase() == "tensor_id" || first.toLowerCase() == "operator_id") {
          childmeta = document.createElement('div');
          specifier = lineread[0] == undefined? '' : lineread[0].trim();
          if (specifier.toLowerCase() == "tensor_id") {
            childmeta.className = "tensor";
          } else {
            childmeta.className = "operator";
          }
          id = lineread[1] == undefined ? '' : lineread[1].trim().slice(0, -1);
          childmeta.innerHTML = JSON.stringify({"id": id});
          
          continue;
        }
        if (first.toLowerCase() == "tensor_style" || first.toLowerCase() == "operator_style") {

          if (line == lines.length - 1) {
            style = lineread[1] == undefined ? '' : lineread[1].trim();
          } else {
            style = lineread[1] == undefined ? '' : lineread[1].trim().slice(0, -1);
          }
          console.log(style);
          var obj = JSON.parse(childmeta.innerHTML);
          obj["style"] = style;
          childmeta.innerHTML = JSON.stringify(obj);
          document.getElementById("list-attributes").appendChild(childmeta);
          
          continue;
        }
        else {
          meta = lineread[1] == undefined ? '' : lineread[1].trim().slice(0, -1);
          var obj = JSON.parse(childmeta.innerHTML);
          if (first.toLowerCase() == "tensor_meta" || first.toLowerCase() == "operator_meta") {
            obj["meta"] = meta;
            
          } else {
            if (first.toLowerCase() == "add_button") {
              if (otherstring !== JSON.stringify({})) {
                obj["button_" + nofb] = otherstring;
                nofb += 1;
              }
              otherstring = JSON.stringify({"id": obj["id"], "class": childmeta.className, "button_name": lineread[1] == undefined ? '' : lineread[1].trim().slice(0, -1)});
            } else {
              if (first.toLowerCase() == "cmd" || first.toLowerCase() == "script") {
                var stri = JSON.parse(otherstring);
                stri[first.toLowerCase()] = lineread[1] == undefined ? '' : lineread[1].trim().slice(0, -1);
                otherstring = JSON.stringify(stri);
              } else {
                if (otherstring !== JSON.stringify({})) {
                  obj["button_" + nofb] = otherstring;
                }
                obj[first] = meta;
              }
            }
          }
          childmeta.innerHTML = JSON.stringify(obj);
          continue;
        } 
      }
      
      if (document.getElementById("list-attributes").children) {
        var lista = document.getElementById("list-attributes");
        var parent_t = document.getElementById("edge-paths");
        var parent_n = document.getElementById("nodes");
        for (var i = 0; i < lista.children.length; i++) {
          if (lista.children[i].className == "tensor") {
            var results = [];
            var len = parent_t.children.length;
            for (var idx = 0; idx <= len; idx++) {
              results.push([idx, parent_t.children[idx]]);
            }
            for (var idx = 0; idx < len; idx++) {
              var obj = JSON.parse(lista.children[i].innerHTML);
              if (results[idx][1].id.split("\n")[1] == obj['id']) {
                //var new_obj = lista.children[i].innerHTML;
                obj["new_id"] = idx;
                obj["tensorname"] = results[idx][1].id;
                lista.children[i].innerHTML = JSON.stringify(obj);
                parent_t.children[idx].style.stroke = obj["style"];
                parent_t.children[idx + 1].innerHTML = '<title>' + obj["meta"].match(/.{1,20}/g).join("\n") + '</title>';
                console.log(lista.children[i].innerHTML);
                break;
              }
            }
          }
          if (lista.children[i].className == "operator") {
            var obj = JSON.parse(lista.children[i].innerHTML);
            var id = obj["id"];
            var idx = 1;
            var counter = 0;
            do{
              var child = parent_n.children[idx];
              if (child.children[3]) {
                counter += 1;
              }
              counter += 1;
              idx += 1;
            } while (idx <= id);
            if (id == 0) {
              counter = 0;
            }
            var new_obj = JSON.parse(lista.children[i].innerHTML);
            new_obj["new_id"] = counter;
            lista.children[i].innerHTML = JSON.stringify(new_obj);
            console.log(lista.children[i].innerHTML);
            var operator = document.getElementById("node-id-" + counter);
            operator.children[0].children[0].innerHTML = '<title>' + obj["meta"].match(/.{1,20}/g).join("\n") + '</title>';
            operator.children[0].children[0].style.fill = obj["style"];
          }
        }
      }
      doubleclick(); 
    }
    reader.readAsText(files[0], 'utf-8');
  };
  input.click();
  window.setInterval(metadata, 100);
  
};

function doubleclick() {
  if (document.getElementById("list-attributes").children) {
    var parent_t = document.getElementById("edge-paths");
    var lista = document.getElementById("list-attributes").children;
    console.log("se verifica");
    console.log(lista.length);
    for (var i = 0; i < lista.length; i++) {
      console.log("se verifica 2")
      var op = JSON.parse(lista[i].innerHTML);
      var id;
      if (lista[i].className == "tensor") {
        id = op["tensorname"];
        parent_t.children[op["new_id"] + 1].addEventListener("dblclick", function() {
          var shell = WScript.CreateObject("WScript.Shell");
          shell.Run("ls");
        })
      } else {
        id = "node-id-" + op["new_id"];
        document.getElementById(id).children[0].addEventListener("dblclick", function() {
          console.log("am apasat pe operator");
        })
      }
      console.log(id);
    }
  }
}

function metadata() {
  if (document.getElementById("list-attributes").children) {
    var lista = document.getElementById("list-attributes");
    var parent_t = document.getElementById("edge-paths");
    var parent_n = document.getElementById("nodes");
    for (var i = 0; i < lista.children.length; i++) {
      var sidebar = document.getElementById("sidebar-content");
      if (sidebar) {
        if (document.getElementById("sidebar-content").children[0]) {
          if (document.getElementById("sidebar-content").children[0].children) {
            var sidebarobj = document.getElementById("sidebar-content").children[0];
            var counter = 0;
            for (var j = 0; j< sidebarobj.children.length; j++) {
              var childd = sidebarobj.children[j];
              if (childd.textContent !== "Metadata" && childd.innerText !== "Metadata" && childd.innerHTML !== "Metadata") {
                  counter += 1;
              }
            }
            var where;
            var element_searched = lista.children[i];
            var inner = JSON.parse(element_searched.innerHTML)
            if (element_searched.className == "operator") {
              where = document.getElementById("node-id-" + inner['new_id']);
            } else {
              where = document.getElementById(inner['tensorname']);
            }
            if (counter == sidebarobj.children.length) {
              if ((element_searched.className == "operator" && where.className['baseVal'] == "node graph-node select") || (element_searched.className == "tensor" && where.className['baseVal'] == "edge-path select")) {
                var keys = Object.keys(inner);
                const childmeta = document.createElement('div');
                childmeta.className = "sidebar-header";
                childmeta.innerText = "Metadata";
                sidebarobj.appendChild(childmeta); 
                for (var i = 0; i < keys.length; i++) {
                  if (keys[i] !== "id" && keys[i] !== "style" && keys[i] !== "new_id" && keys[i] !== "tensorname" && keys[i].slice(0, 7) !== "button_") {
                    var newchild = document.createElement('div');
                    newchild.className = "sidebar-item";
                    var newchild_2 = document.createElement('div');
                    newchild_2.className = "sidebar-item-name";
                    var inputc = document.createElement('input');
                    inputc.type="text";
                    inputc.value=keys[i];
                    inputc.title=keys[i];
                    inputc.readonly="true";

                    var newchild_3 = document.createElement('div');
                    newchild_3.className = "sidebar-item-value-list";

                    var newchild_4 = document.createElement('div');
                    newchild_4.className = "sidebar-item-value";

                    var newchild_5 = document.createElement('div');
                    newchild_5.className = "sidebar-item-value-expander";
                    newchild_5.innerText = '+';

                    var newchild_6 = document.createElement('div');
                    newchild_6.className = "sidebar-item-value-line";
                    newchild_6.style="cursor: pointer;";

                    var newchild_7 = document.createElement('span');
                    newchild_7.className = "sidebar-item-value-line-content";
                    newchild_7.innerHTML = `<b>` + inner[keys[i]] + `</b>`;
                    
                    newchild_6.append(newchild_7);
                    newchild_4.append(newchild_5);
                    newchild_4.append(newchild_6);
                    newchild_3.append(newchild_4);
                    newchild_2.append(inputc);
                    newchild.appendChild(newchild_2);
                    newchild.appendChild(newchild_3);
                    sidebarobj.appendChild(newchild);
                  }
                }
                for (var i = 0; i < keys.length; i++) {
                  if (keys[i].slice(0, 7) == "button_") {
                    console.log(keys[i]);
                    var stringnew = JSON.parse(inner[keys[i]]);
                    console.log(stringnew);
                    console.log(inner["id"]);
                    console.log(stringnew["id"]);
                    if (inner["id"] == stringnew["id"]) {
                      console.log("am gasit buton");
                      var newchild = document.createElement('button');
                      newchild.type = "button";
                      newchild.name = stringnew["button_name"];
                      newchild.innerHTML = stringnew["cmd"];
                      newchild.style = "position: relative;";
                      sidebarobj.appendChild(newchild);
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
export default functie;