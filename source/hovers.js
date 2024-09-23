class Req {
  constructor() {}
static async _request(url) {
  let timeout = 0.25;
  let callback = 0;
  return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.responseType = 'arraybuffer';
      if (timeout) {
          request.timeout = timeout;
      }
      const progress = (value) => {
          if (callback) {
              callback(value);
          }
      };
      request.onload = () => {
          progress(0);
          if (request.status === 200) {
              let value = null;
              if (request.responseType === 'arraybuffer') {
                var eroare = JSON.stringify(request);
                var resp = request.response;
                const decoder = new TextDecoder();
                var textrec = decoder.decode(resp);
                if (textrec.slice(0, 8) == "command ") {
                  alert("Response for command is " + textrec.slice(8, textrec.length - 1));
                }
                else if (textrec == "comanda") {
                  alert("Script executed");
                }
              } else {
                  value = request.responseText;
              }
              resolve(value);
          } else {
            if (request.status === 202) {
              var eroare = JSON.stringify(request);
              const error = new Error(`The web request failed with status code '${eroare}'.`);
              error.context = url;
              reject(error);
            }
          }
      };
      request.onerror = () => {
          progress(0);
          const error = new Error(`The web request failed.`);
          error.context = url;
          reject(error);
      };
      request.ontimeout = () => {
          progress(0);
          request.abort();
          const error = new Error('The web request timed out.', 'timeout', url);
          error.context = url;
          reject(error);
      };
      request.onprogress = (e) => {
          if (e && e.lengthComputable) {
              progress(e.loaded / e.total * 100);
          }
      };
      request.open('GET', url, true);
      request.send();
  });
};

static functie() {
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
        if (!lines[line].startsWith("#") && lines[line].trim().length !== 0) {
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
          id = lineread[1] == undefined ? '' : lineread[1].trim();
          childmeta.innerHTML = JSON.stringify({"id": id});
          continue;
        }
        if (first.toLowerCase() == "tensor_style" || first.toLowerCase() == "operator_style") {

          if (line == lines.length - 1) {
            style = lineread[1] == undefined ? '' : lineread[1].trim();
          } else {
            style = lineread[1] == undefined ? '' : lineread[1].trim();
          }
          var obj = JSON.parse(childmeta.innerHTML);
          obj["style"] = style;
          childmeta.innerHTML = JSON.stringify(obj);
          document.getElementById("list-attributes").appendChild(childmeta);
          continue;
        }
        else {
          meta = lineread[1] == undefined ? '' : lineread[1].trim();
          var obj = JSON.parse(childmeta.innerHTML);
          if (first.toLowerCase() == "tensor_meta" || first.toLowerCase() == "operator_meta") {
            obj["meta"] = meta;
          }
          else if (first.toLowerCase() == "tensor_ondblclick_script" || first.toLowerCase() == "tensor_ondblclick_command" || first.toLowerCase() == "operator_ondblclick_script" || first.toLowerCase() == "operator_ondblclick_command") {
            obj[first.toLowerCase()] = meta;
          } else {
            if (first.toLowerCase() == "tensor_button" || first.toLowerCase() == "operator_button") {
              if (otherstring !== JSON.stringify({})) {
                obj["button_" + nofb] = otherstring;
                nofb += 1;
              }
              otherstring = JSON.stringify({"id": obj["id"], "class": childmeta.className, "button_name": lineread[1] == undefined ? '' : lineread[1].trim()});
            } else {
              if (first.toLowerCase() == "tensor_button_cmd" || first.toLowerCase() == "tensor_button_script" || first.toLowerCase() == "operator_button_cmd" || first.toLowerCase() == "operator_button_script") {
                var stri = JSON.parse(otherstring);
                stri[first.toLowerCase()] = lineread[1] == undefined ? '' : lineread[1].trim();
                otherstring = JSON.stringify(stri);
              } else {
                if (otherstring !== JSON.stringify({})) {
                  obj["button_" + nofb] = otherstring;
                }
                otherstring = JSON.stringify({});
                obj[first] = meta;
              }
            }
          }
          childmeta.innerHTML = JSON.stringify(obj);
          continue;
        } 
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
                obj["new_id"] = idx;
                obj["tensorname"] = results[idx][1].id;
                lista.children[i].innerHTML = JSON.stringify(obj);
                parent_t.children[idx].style.stroke = obj["style"];
                parent_t.children[idx + 1].innerHTML = '<title>' + obj["meta"].match(/.{1,20}/g).join("\n") + '</title>';
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
            var operator = document.getElementById("node-id-" + counter);
            operator.children[0].children[0].innerHTML = '<title>' + obj["meta"].match(/.{1,20}/g).join("\n") + '</title>';
            operator.children[0].children[0].style.fill = obj["style"];
          }
        }
      }
      Req.doubleclick(); 
    }
    reader.readAsText(files[0], 'utf-8');
  };
  input.click();
  window.setInterval(Req.metadata, 100);

};

static doubleclick() {
  if (document.getElementById("list-attributes").children) {
    var parent_t = document.getElementById("edge-paths");
    var parent_n = document.getElementById("nodes");
    var lista = document.getElementById("list-attributes").children;
    for (var i = 0; i < lista.length; i++) {
      if (lista[i].className === "tensor") {
        var op = JSON.parse(lista[i].innerHTML);
        var id = op["tensorname"];
        lista[i].id = op["new_id"] + 1;
      } else {
        var op = JSON.parse(lista[i].innerHTML);
        var id = "node-id-" + op["new_id"];
        lista[i].id = id;
      }
    }
    for (var i = 0; i < lista.length; i++) {
      if (lista[i].className == "tensor") {
        var variable = lista[i].id;
        (parent_t.children[variable]).addEventListener("dblclick", function(e) {
          var idx = 0;
          var op;
          for (var i = 0; i < parent_t.children.length; i++) {
            if (parent_t.children[i].isEqualNode(e.srcElement)) {
              idx = i;
              break;
            }
          }
          for (var i = 0; i < lista.length; i++) {
            if (lista[i].id == idx) {
              op = JSON.parse(lista[i].innerHTML);
              break;
            }
          }
          if (op["dblclick_cmd"]) {
            var result = Req._request("/command_" + op["dblclick_cmd"]);
          }
          if (op["dblclick_script"]) {
            var prefix = op["dblclick_script"].slice(0, 2);
            var script = op["dblclick_script"].slice(3, -1) + ");";
            if (prefix === "js") {
              eval(script);
            } else {
              var result = Req._request("/scriptforadding_" + script);
            }
          }
        })
      } else {
        var variable = lista[i].id;
        (document.getElementById(variable)).addEventListener("dblclick", function(g) {
          var idx = 0;
          var op;
          for (var i = 0; i < parent_n.children.length; i++) {
            if ((parent_n.children[i].children[0].children[0]).isEqualNode(g.srcElement) || (parent_n.children[i].children[0].children[1]).isEqualNode(g.srcElement)) {
              idx = parent_n.children[i].id;
              break;
            }
          }
          for (var i = 0; i < lista.length; i++) {
            if (lista[i].id == idx) {
              op = JSON.parse(lista[i].innerHTML);
              break;
            }
          }
          if (op["dblclick_cmd"]) {
            var result = Req._request("/command_" + op["dblclick_cmd"]);
          }
          if (op["dblclick_script"]) {
            var prefix = op["dblclick_script"].slice(0, 2);
            var script = op["dblclick_script"].slice(3, -1) + ");";
            if (prefix === "js") {
              eval(script);
            } else {
              var result = Req._request("/scriptforadding_" + script);
            }
          }
        })
      }
    }
  }
}

static metadata() {
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
                  if (keys[i] !== "id" && keys[i] !== "style" && keys[i] !== "new_id" && keys[i] !== "tensorname" && keys[i].slice(0, 7) !== "button_" && keys[i] !== "dblclick_script" && keys[i] !== "dblclick_cmd" && keys[i] != "cmd" && keys[i] !== "script") {
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
                    var stringnew = JSON.parse(inner[keys[i]]);
                    if (inner["id"] == stringnew["id"]) {
                      var newchild = document.createElement('button');
                      newchild.type = "button";
                      newchild.textContent = stringnew["button_name"];
                      newchild.style = "position: relative;";
                      if (stringnew["cmd"] && stringnew["script"]) {
                        newchild.value = JSON.stringify({"cmd": stringnew["cmd"], "script": stringnew["script"]});
                      } else {
                        if (stringnew["cmd"]) {
                          newchild.value += JSON.stringify({"cmd": stringnew["cmd"]});
                        } else if (stringnew["script"]) {
                          newchild.value += JSON.stringify({"script": stringnew["script"]});
                        } else {
                          newchild.value += JSON.stringify({});
                        }
                      }
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
  if (document.getElementById("list-attributes").children) {
    if (document.getElementById("sidebar-content")) {
      if (document.getElementById("sidebar-content").children[0]) {
        if (document.getElementById("sidebar-content").children[0].children) {
          var sidebarobj = document.getElementById("sidebar-content").children[0];
          for (var i = 0; i < sidebarobj.children.length; i++) {
            if (sidebarobj.children[i].type == "button") {
              if (sidebarobj.children[i].getAttribute('listener') !== 'true') {
                sidebarobj.children[i].setAttribute('listener', 'true');
                sidebarobj.children[i].addEventListener('click', function(h) {
                  var value = JSON.parse(h.target.value);
                  if (value["cmd"]) {
                    var result = Req._request("/command_" + value["cmd"]);
                  }
                  if (value["script"]) {
                    var prefix = value["script"].slice(0, 2);
                    var script = value["script"].slice(3, -1) + ");";
                    if (prefix === "js") {
                      eval(script);
                    } else {
                      var result = Req._request("/scriptforadding_" + script);
                    }
                  }
                });
              }
            }
          }
        }
      }
    }
  }
  
}
}

export default Req;