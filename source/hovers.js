import { stringify } from 'querystring';

import { execSync } from 'child_process';
class Req {
  file_added = 0;
  constructor() {}

  static solve_bugs() {
    var root = document.getElementById("list-modified");
    if (root.length !== 0)
    {
      for (var i = 0; i < root.children.length; i++) {
          var child = JSON.parse(root.children[i].innerHTML);
          var id_child = child["id"];
          if (root.children[i].className == "tensor") {
              var list_t = document.getElementById("edge-paths");
              for (var j = 0; j < list_t.children.length; j++) {
                  if (list_t.children[j].id.split("\n")[1] == id_child) {
                      if (child["style"] && list_t.children[j].hasAttribute("style")) {
                          list_t.children[j].removeAttribute("style");
                      }
                      if (child["hover"] && list_t.children[j + 1].innerHTML !== '') {
                          list_t.children[j + 1].innerHTML = '';
                      }
                  }
              }
          } else {
              var parent_n = document.getElementById("nodes");
              var idx = 1;
              var counter = 0;
              do{
                  var child = parent_n.children[idx];
                  if (child.children[3]) {
                  counter += 1;
                  }
                  counter += 1;
                  idx += 1;
              } while (idx <= id_child);
              if (id_child == 0) {
                  counter = 0;
              }
              var operator = document.getElementById("node-id-" + counter);
              if (operator) {
                if (operator.children) {
                  if (operator.children[0]) {
                    if (operator.children[0].children) {
                      if (operator.children[0].children[0]) {
                        if (child["hover"] && operator.children[0].children[0].innerHTML !== '') {
                          operator.children[0].children[0].innerHTML = '';
                        } if (child["style"] && operator.children[0].children[0].hasAttribute("style")) {
                            operator.children[0].children[0].removeAttribute("style");
                        }
                      }
                    }
                  }
                }
              }
          }
      }
    }
    document.getElementById("list-attributes").innerHTML = '';
    document.getElementById("list-modified").innerHTML = '';
    Req.file_added = 0;
  }

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
                  alert(textrec);
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
      let files = Array.from(input.files);
      const reader = new FileReader();
      reader.onload = function() {
        try {
        Req.file_added = 1;
        const content = reader.result;
        var lines = content.split('\n');
        var id, meta, style, specifier;
        var childmeta = 0;
        var another = 0;
        var nofb = 0;
        var otherstring = JSON.stringify({});
        var optionskeys = ["operator_onmouseover_image_posx", "operator_onmouseover_image_posy", "tensor_onmouseover_image_posx", "tensor_onmouseover_image_posy", "tensor_id", "operator_id", "tensor_style", "operator_style", "tensor_ondblclick_script", "tensor_ondblclick_command", "operator_ondblclick_script", "operator_ondblclick_command", "tensor_onmouseover_text", "operator_onmouseover_text", "tensor_onmouseover_image", "operator_onmouseover_image", "tensor_meta_key", "operator_meta_key", "tensor_meta_val", "operator_meta_val", "tensor_button", "operator_button", "tensor_button_command", "tensor_button_script", "operator_button_command", "operator_button_script", "tensor_onmouseover_image_dimx", "tensor_onmouseover_image_dimy", "operator_onmouseover_image_dimx", "operator_onmouseover_image_dimy"];
        var index = -1;
        var img = JSON.stringify({});
        var nofimg = 0

        for (var line = 0; line < lines.length; line++) {
          if (!lines[line].startsWith("#") && lines[line].trim().length !== 0) {
            var lineread = lines[line].split(/:(.+)/);
            var first = lineread[0] == undefined? '' : lineread[0].trim();
            console.log(first.toLowerCase());
            if (index == -1) {
              if (first.toLowerCase() !== "operator_id" && first.toLowerCase() !== "tensor_id") {
                Req.solve_bugs();
                throw "Invalid model metadata file format! The first line must be operator_id or tensor_id";
              }
            }
            index = 0;
            if (!optionskeys.includes(first.toLowerCase())) {
              Req.solve_bugs();
              throw "Invalid model metadata file format! You provided an invalid key: " + first.toLowerCase();
            }
            if (first.toLowerCase() == "tensor_id" || first.toLowerCase() == "operator_id") {
              if (otherstring !== JSON.stringify({})) {
                var objeect = document.getElementById("list-attributes");
                for (var i = 0; i < objeect.children.length; i++) {
                  if ((JSON.parse(objeect.children[i].innerHTML))['id'] == JSON.parse(otherstring)['id']) {
                    var inner = JSON.parse(objeect.children[i].innerHTML);
                    inner["button_" + nofb] = otherstring;
                    objeect.children[i].innerHTML = JSON.stringify(inner);
                    otherstring = JSON.stringify({});
                    nofb += 1;
                    break;
                  }
                }
              }
              if (img !== JSON.stringify({})) {
                var objeect = document.getElementById("list-attributes");
                  for (var i = 0; i < objeect.children.length; i++) {
                    if ((JSON.parse(objeect.children[i].innerHTML))['id'] == JSON.parse(img)['id']) {
                      var inner = JSON.parse(objeect.children[i].innerHTML);
                      inner["img_" + nofimg] = img;
                      objeect.children[i].innerHTML = JSON.stringify(inner);
                      img = JSON.stringify({});
                      nofimg += 1;
                      break;
                  }
                }
              }
              if (childmeta !== 0) {
                document.getElementById("list-attributes").appendChild(childmeta);
                document.getElementById("list-modified").appendChild(another);
                childmeta = 0;
                another = 0;
              }
              childmeta = document.createElement('div');
              another = document.createElement('div');
              specifier = lineread[0] == undefined? '' : lineread[0].trim();
              if (specifier.toLowerCase() == "tensor_id") {
                childmeta.className = "tensor";
                another.className = "tensor";
              } else {
                childmeta.className = "operator";
                another.className = "operator";
              }
              id = lineread[1] == undefined ? '' : lineread[1].trim();
              if (!(!isNaN(parseFloat(id)) && isFinite(id))) {
                Req.solve_bugs();
                throw "Invalid model metadata file format! Id must be a number";
              }
              childmeta.innerHTML = JSON.stringify({"id": id});
              another.innerHTML = JSON.stringify({"id": id});
              continue;
            }
            if (first.toLowerCase() == "tensor_style" || first.toLowerCase() == "operator_style") {
              if ((childmeta.className == "tensor" && first.toLowerCase() == "operator_style") || (childmeta.className == "operator" && first.toLowerCase() == "tensor_style")) {
                Req.solve_bugs();
                throw "Invalid model metadata file format! tensor_style must correspond to tensor and operator_style to operator";
              }
              style = lineread[1] == undefined ? '' : lineread[1].trim();
              var obj = JSON.parse(childmeta.innerHTML);
              obj["style"] = style;
              childmeta.innerHTML = JSON.stringify(obj);
              var obj2 = JSON.parse(another.innerHTML);
              obj2["style"] = style;
              another.innerHTML = JSON.stringify(obj2);
              continue;
            }
            else {
              meta = lineread[1] == undefined ? '' : lineread[1].trim();
              var obj = JSON.parse(childmeta.innerHTML);
              var obj2 = JSON.parse(another.innerHTML);
              if (first.toLowerCase() == "tensor_ondblclick_script" || first.toLowerCase() == "tensor_ondblclick_command" || first.toLowerCase() == "operator_ondblclick_script" || first.toLowerCase() == "operator_ondblclick_command") {
                if (((first.toLowerCase() == "tensor_ondblclick_script" || first.toLowerCase() == "tensor_ondblclick_command") && childmeta.className == "operator") || ((first.toLowerCase() == "operator_ondblclick_script" || first.toLowerCase() == "operator_ondblclick_command") && childmeta.className == "tensor")) {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! tensor_ondblclick.. must correspond to tensor and operator_ondblclick... to operator";
                }
                obj[first.toLowerCase()] = meta;
              } else if (first.toLowerCase() == "tensor_onmouseover_text" || first.toLowerCase() == "operator_onmouseover_text") {
                if ((first.toLowerCase() == "tensor_onmouseover_text" && childmeta.className == "operator") || (first.toLowerCase() == "operator_onmouseover_text" && childmeta.className == "tensor")) {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! tensor_onmouseover_text must correspond to tensor and operator_onmouseover_text to operator";
                }
                obj["hover"] = meta;
                obj2["hover"] = meta;
              }
              else if (first.toLowerCase() == "tensor_onmouseover_image" || first.toLowerCase() == "operator_onmouseover_image") {
                if ((first.toLowerCase() == "tensor_onmouseover_image" && childmeta.className == "operator") || (first.toLowerCase() == "operator_onmouseover_image" && childmeta.className == "tensor")) {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! tensor_onmouseover_image must correspond to tensor and operator_onmouseover_image to operator";
                }
                if (img !== JSON.stringify({})) {
                  obj["img_" + nofimg] = img;
                  nofimg += 1;
                }
                img = JSON.stringify({"id": obj["id"], "class": childmeta.className, "img_link": meta});
                obj2["hover_image"] = meta;
              }
              else if (first.toLowerCase() == "tensor_meta_key" || first.toLowerCase() == "operator_meta_key") {
                if ((first.toLowerCase() == "tensor_meta_key" && childmeta.className == "operator") || (first.toLowerCase() == "operator_meta_key" && childmeta.className == "tensor")) {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! tensor_meta_key must correspond to tensor and operator_meta_key to operator";
                }
                var k = 1;
                while (lines[line + k].startsWith("#") || lines[line + k].trim().length == 0) {
                  k += 1;
                }
                var linereadsecond = lines[line + k].split(":");
                var second = linereadsecond[0] == undefined? '' : linereadsecond[0].trim();
                if (second.toLowerCase() == "tensor_meta_val" || second.toLowerCase() == "operator_meta_val") {
                  if ((first.toLowerCase() == "tensor_meta_val" && childmeta.className == "operator") || (first.toLowerCase() == "operator_meta_val" && childmeta.className == "tensor")) {
                    Req.solve_bugs();
                    throw "Invalid model metadata file format! tensor_meta_val must correspond to tensor and operator_meta_val to operator";
                  }
                  var value = linereadsecond[1].trim();
                  var key = lineread[1].trim();
                  obj[key] = value;
                  obj2["metadata"] = "metadata";
                } else {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! meta_val must be after meta_key";
                }
              }
              else if (first.toLowerCase() == "tensor_meta_val" || first.toLowerCase() == "operator_meta_val" ) {
                var k = -1;
                while (lines[line + k].startsWith("#") || lines[line + k].trim().length == 0) {
                  k -= 1;
                }
                var linereadsecond = lines[line + k].split(":");
                var second = linereadsecond[0] == undefined? '' : linereadsecond[0].trim();
                if ((second.toLowerCase() == "tensor_meta_key" && first.toLowerCase() == "tensor_meta_val") || (second.toLowerCase() == "operator_meta_key" && first.toLowerCase() == "operator_meta_val")) {
                  continue;
                } else if ((second.toLowerCase() == "tensor_meta_key" && first.toLowerCase() == "operator_meta_val") || (second.toLowerCase() == "operator_meta_key" && first.toLowerCase() == "tensor_meta_val")) {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! both key and value must be part of tensor or operator";
                } else {
                  Req.solve_bugs();
                  throw "Invalid model metadata file format! meta_val must be after meta_key";
                }
              }
              else {
                if (first.toLowerCase() == "tensor_button" || first.toLowerCase() == "operator_button") {
                  if ((childmeta.className == "operator" && first.toLowerCase() == "tensor_button") || (childmeta.className == "tensor" && first.toLowerCase() == "operator_button")) {
                    Req.solve_bugs();
                    throw "Invalid model metadata file format! tensor_button must correspond to tensor and operator_button to operator";
                  }
                  var k = 1;
                  while (lines[line + k].startsWith("#") || lines[line + k].trim().length == 0) {
                    k += 1;
                  }
                  var linereadsecond = lines[line + k].split(":");
                  var second = linereadsecond[0] == undefined? '' : linereadsecond[0].trim();
                  if ((first.toLowerCase() == "tensor_button" && second.toLowerCase() == "tensor_button") || (first.toLowerCase() == "operator_button" && second.toLowerCase() == "operator_button")) {
                    Req.solve_bugs();
                    throw "Invalid model metadata file format! You can't have empty buttons";
                  }
                  if (otherstring !== JSON.stringify({})) {
                    obj["button_" + nofb] = otherstring;
                    nofb += 1;
                  }
                  otherstring = JSON.stringify({"id": obj["id"], "class": childmeta.className, "button_name": lineread[1] == undefined ? '' : lineread[1].trim()});
                  obj2["buttons"] = "buttons";
                } else {
                  if (first.toLowerCase() == "tensor_button_command" || first.toLowerCase() == "tensor_button_script" || first.toLowerCase() == "operator_button_command" || first.toLowerCase() == "operator_button_script") {
                    if ((childmeta.className == "tensor" && (first.toLowerCase() == "operator_button_command" || first.toLowerCase() == "operator_button_script")) || (childmeta.className == "operator" && (first.toLowerCase() == "tensor_button_command" || first.toLowerCase() == "tensor_button_script"))) {
                      Req.solve_bugs();
                      throw "Invalid model metadata file format! tensor_button.. must correspond to tensor and operator_button.. to operator";
                    }
                    var k = -1;
                    while (lines[line + k].startsWith("#") || lines[line + k].trim().length == 0) {
                      k -= 1;
                    }
                    var linereadsecond = lines[line + k].split(":");
                    var second = linereadsecond[0] == undefined? '' : linereadsecond[0].trim();
                    if ((first.toLowerCase() == "tensor_button_command" && second.toLowerCase() == "tensor_button_command") || (first.toLowerCase() == "tensor_button_script" && second.toLowerCase() == "tensor_button_script") || (first.toLowerCase() == "operator_button_command" && second.toLowerCase() == "operator_button_command") || (first.toLowerCase() == "operator_button_script" && second.toLowerCase() == "operator_button_script")) {
                      Req.solve_bugs();
                      throw "A button can't execute two scripts or two commands at the same time";
                    }
                    if (first.toLowerCase() == "tensor_button_command") {
                      if (second.toLowerCase() !== "tensor_button_script" && second.toLowerCase() !== "tensor_button") {
                        Req.solve_bugs();
                        throw "The format of the data is wrong: the construction of the button should be tensor_button, tensor_button_command, tensor_button_script or tensor_button, tensor_button_script, tensor_button_command";
                      }
                    }
                    if (first.toLowerCase() == "tensor_button_script") {
                      if (second.toLowerCase() !== "tensor_button_command" && second.toLowerCase() !== "tensor_button") {
                        Req.solve_bugs();
                        throw "The format of the data is wrong: the construction of the button should be tensor_button, tensor_button_command, tensor_button_script or tensor_button, tensor_button_script, tensor_button_command";
                      }
                    }
                    if (first.toLowerCase() == "operator_button_command") {
                      if (second.toLowerCase() !== "operator_button_script" && second.toLowerCase() !== "operator_button") {
                        Req.solve_bugs();
                        throw "The format of the data is wrong: the construction of the button should be operator_button, operator_button_command, operator_button_script or operator_button, operator_button_script, operator_button_command";
                      }
                    }
                    if (first.toLowerCase() == "operator_button_command") {
                      if (second.toLowerCase() !== "operator_button_script" && second.toLowerCase() !== "operator_button") {
                        Req.solve_bugs();
                        throw "The format of the data is wrong: the construction of the button should be operator_button, operator_button_command, operator_button_script or operator_button, operator_button_script, operator_button_command";
                      }
                    }
                    var stri = JSON.parse(otherstring);
                    stri[first.toLowerCase()] = lineread[1] == undefined ? '' : lineread[1].trim();
                    otherstring = JSON.stringify(stri);
                  } 
                  else {
                    if (first.toLowerCase() == "tensor_onmouseover_image_dimx" || first.toLowerCase() == "tensor_onmouseover_image_dimy" || first.toLowerCase() == "operator_onmouseover_image_dimx" || first.toLowerCase() == "operator_onmouseover_image_dimy" || first.toLowerCase() == "tensor_onmouseover_image_posx" || first.toLowerCase() == "tensor_onmouseover_image_posy" || first.toLowerCase() == "operator_onmouseover_image_posx" || first.toLowerCase() == "operator_onmouseover_image_posy") {
                      var strin = JSON.parse(img);
                      strin[first.toLowerCase()] = lineread[1] == undefined ? '' : lineread[1].trim();
                      img = JSON.stringify(strin);
                    } else {
                      if (otherstring !== JSON.stringify({})) {
                        obj["button_" + nofb] = otherstring;
                      }
                      if (img !== JSON.stringify({})) {
                        obj["img_" + nofimg] = img;
                      }
                      otherstring = JSON.stringify({});
                      img = JSON.stringify({});
                      document.getElementById("list-attributes").innerHTML = '';
                      throw "You provided an invalid key: " + first.toLowerCase();
                    }
                  }
                }
              }
              childmeta.innerHTML = JSON.stringify(obj);
              another.innerHTML = JSON.stringify(obj2);
              continue;
            } 
          }
        }
        if (childmeta !== 0) {
          document.getElementById("list-attributes").appendChild(childmeta);
          document.getElementById("list-modified").appendChild(another);
        }
        if (otherstring !== JSON.stringify({})) {
          var objeect = document.getElementById("list-attributes");
            for (var i = 0; i < objeect.children.length; i++) {
              if ((JSON.parse(objeect.children[i].innerHTML))['id'] == JSON.parse(otherstring)['id']) {
                var inner = JSON.parse(objeect.children[i].innerHTML);
                inner["button_" + nofb] = otherstring;
                objeect.children[i].innerHTML = JSON.stringify(inner);
                otherstring = JSON.stringify({});
                nofb += 1;
                break;
            }
          }
        }
        if (img !== JSON.stringify({})) {
          var objeect = document.getElementById("list-attributes");
            for (var i = 0; i < objeect.children.length; i++) {
              if ((JSON.parse(objeect.children[i].innerHTML))['id'] == JSON.parse(img)['id']) {
                var inner = JSON.parse(objeect.children[i].innerHTML);
                inner["img_" + nofimg] = img;
                objeect.children[i].innerHTML = JSON.stringify(inner);
                img = JSON.stringify({});
                nofimg += 1;
                break;
            }
          }
        }
        if (document.getElementById("list-attributes").children) {
          var lista = document.getElementById("list-attributes");
          var lista_m = document.getElementById("list-modified");
          var parent_t = document.getElementById("edge-paths");
          var parent_n = document.getElementById("nodes");
          for (var i = 0; i < lista.children.length; i++) {
            if (lista.children[i].className == "tensor") {
              var results = [];
              var len = parent_t.children.length;
              for (var idx = 0; idx <= len; idx++) {
                results.push([idx, parent_t.children[idx]]);
              }
              var flag = 0;
              for (var idx = 0; idx < len; idx++) {
                var obj = JSON.parse(lista.children[i].innerHTML);
                var obj2 = JSON.parse(lista_m.children[i].innerHTML);
                if (results[idx][1].id.split("\n")[1] == obj['id']) {
                  obj["new_id"] = idx;
                  obj["tensorname"] = results[idx][1].id;
                  obj2["new_id"] = idx;
                  obj2["tensorname"] = results[idx][1].id;
                  lista.children[i].innerHTML = JSON.stringify(obj);
                  if (obj["style"]) {
                    parent_t.children[idx].style.stroke = obj["style"];
                  }
                  flag = 1;
                  if (obj["hover"]) {
                    parent_t.children[idx + 1].innerHTML = '<title>' + obj["hover"].match(/.{1,20}/g).join("\n") + '</title>';
                  }
                  var keys = Object.keys(obj);
                  for (var j = 0; j < keys.length; j++) {
                    if (keys[j].startsWith("img_")) {
                      var obje = JSON.parse(obj[keys[j]]);
                      var oImg = document.createElement('img');
                      oImg.src = obje["img_link"];
                      oImg.style.width = obje["tensor_onmouseover_image_dimx"];
                      oImg.style.height = obje["tensor_onmouseover_image_dimy"];
                      oImg.style.display = 'none';
                      oImg.style.position = 'absolute';
                      oImg.style.top = obje["tensor_onmouseover_image_posx"];
                      oImg.style.left = obje["tensor_onmouseover_image_posy"];
                      oImg.id = "tensor-image-" + idx + "_" + keys[j];
                      oImg.className = "put-image-on-hover";
                      document.body.appendChild(oImg);
                    }
                  }
                  break;
                }
              }
              if (flag == 0) {
                Req.solve_bugs();
                throw "Index of tensor not found :" + obj['id'];
              }
            }
            if (lista.children[i].className == "operator") {
              var obj = JSON.parse(lista.children[i].innerHTML);
              var obj2 = JSON.parse(lista_m.children[i].innerHTML);
              var id = obj["id"];
              var idx = 1;
              var counter = 0;
              var flag = 0;
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
              obj2["new_id"] = counter;
              lista.children[i].innerHTML = JSON.stringify(new_obj);
              lista_m.children[i].innerHTML = JSON.stringify(obj2);
              var operator = document.getElementById("node-id-" + counter);
              if (!operator) {
                Req.solve_bugs();
                throw "Index of operator not found :" + obj["id"];
              }
              if (obj["hover"]) {
                operator.children[0].children[0].innerHTML = '<title>' + obj["hover"].match(/.{1,20}/g).join("\n") + '</title>';
              }

              var keys = Object.keys(obj);
                  for (var j = 0; j < keys.length; j++) {
                    if (keys[j].startsWith("img_")) {
                      var obje = JSON.parse(obj[keys[j]]);
                      var oImg = document.createElement('img');
                      oImg.src = obje["img_link"];
                      oImg.style.width = obje["operator_onmouseover_image_dimx"];
                      oImg.style.height = obje["operator_onmouseover_image_dimy"];
                      oImg.style.display = 'none';
                      oImg.style.position = 'absolute';
                      oImg.style.top = obje["operator_onmouseover_image_posx"];
                      oImg.style.left = obje["operator_onmouseover_image_posy"];
                      oImg.id = "image-node-id-" + counter + "_" + keys[j];
                      oImg.className = "put-image-on-hover";
                      document.body.appendChild(oImg);
                    }
                  }
              if (obj["style"]) {
                operator.children[0].children[0].style.fill = obj["style"];
              }
            }
          }
        }
        Req.doubleclick();
        window.setInterval(Req.metadata, 100);
      } catch(Err) {
        alert(Err);
      }
      }
      reader.readAsText(files[0], 'utf-8');
    };
    input.click();
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
            if (op["tensor_ondblclick_command"]) {
              try {
                const execSync = require('child_process').execSync;
                const output = execSync(op["tensor_ondblclick_command"], {shell: true});
              } catch(Err) {
                alert("Error when running the command " + op["tensor_ondblclick_command"] + ": " + Err.message);
              }
            }
            if (op["tensor_ondblclick_script"]) {
              try {
                eval(op["tensor_ondblclick_script"]);
              } catch(Err) {
                alert("Error when running the script " + op["tensor_ondblclick_script"] + ": " + Err.message);
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
            if (op["operator_ondblclick_command"]) {
              try {
                const execSync = require('child_process').execSync;
                  const output = execSync(op["operator_ondblclick_command"], {shell: true});
              } catch(Err) {
                alert("Error when running the command " + op["operator_ondblclick_command"] + ": " + Err.message);
              }
            }
            if (op["operator_ondblclick_script"]) {
                try {
                  eval(op["operator_ondblclick_script"]);
                } catch(Err) {
                  alert("Error when running the script " + op["operator_ondblclick_script"] + ": " + Err.message);
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
        var element_searched = lista.children[i];
        var inner = JSON.parse(element_searched.innerHTML)
        var where;
        if (element_searched.className == "operator") {
          where = document.getElementById("node-id-" + inner['new_id']);
        } else {
          where = document.getElementById(inner['tensorname']);
        }
        if (where.getAttribute('listener') !== 'true') {
          where.setAttribute('listener', 'true');
        }
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
              if (counter == sidebarobj.children.length) {
                if ((element_searched.className == "operator" && where.className['baseVal'] == "node graph-node select") || (element_searched.className == "tensor" && where.className['baseVal'] == "edge-path select")) {
                  var keys = Object.keys(inner);
                  const childmeta = document.createElement('div');
                  childmeta.className = "sidebar-header";
                  childmeta.innerText = "Metadata";
                  sidebarobj.appendChild(childmeta);
                  for (var i = 0; i < keys.length; i++) {
                    if (keys[i] !== "id" && keys[i] !== "style" && keys[i] !== "new_id" && keys[i] !== "tensorname" && keys[i].slice(0, 7) !== "button_" && keys[i] !== "tensor_ondblclick_script" && keys[i] !== "operator_ondblclick_script" && keys[i] !== "tensor_ondblclick_command" && keys[i] !== "operator_ondblclick_command" && keys[i] !== "hover" && keys[i] !== "hover_image") {
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
                  var repeated = 0;
                  for (var i = 0; i < keys.length; i++) {
                    if (keys[i].slice(0, 7) == "button_") {
                      if (repeated == 0) {
                        const childmeta = document.createElement('div');
                        childmeta.className = "sidebar-header";
                        childmeta.innerText = "Actions";
                        sidebarobj.appendChild(childmeta);
                        repeated = 1;
                      }
                      var stringnew = JSON.parse(inner[keys[i]]);
                      if (inner["id"] == stringnew["id"]) {
                        var newchild = document.createElement('button');
                        newchild.type = "button";
                        newchild.textContent = stringnew["button_name"];
                        newchild.style = "display: block; margin: 10px; background-color:#99c2ff; color:black; border-radius: 8px;";
                        if (stringnew["class"] == "tensor") {
                          if (stringnew["tensor_button_command"] && stringnew["tensor_button_script"]) {
                            newchild.value = JSON.stringify({"cmd": stringnew["tensor_button_command"], "script": stringnew["tensor_button_script"]});
                          } else {
                            if (stringnew["tensor_button_command"]) {
                              newchild.value += JSON.stringify({"cmd": stringnew["tensor_button_command"]});
                            } else if (stringnew["tensor_button_script"]) {
                              newchild.value += JSON.stringify({"script": stringnew["tensor_button_script"]});
                            } else {
                              newchild.value += JSON.stringify({});
                            }
                          }
                        }
                        if (stringnew["class"] == "operator") {
                          if (stringnew["operator_button_command"] && stringnew["operator_button_script"]) {
                            newchild.value = JSON.stringify({"cmd": stringnew["operator_button_command"], "script": stringnew["operator_button_script"]});
                          } else {
                            if (stringnew["operator_button_command"]) {
                              newchild.value += JSON.stringify({"cmd": stringnew["operator_button_command"]});
                            } else if (stringnew["operator_button_script"]) {
                              newchild.value += JSON.stringify({"script": stringnew["operator_button_script"]});
                            } else {
                              newchild.value += JSON.stringify({});
                            }
                          }
                        }
                        sidebarobj.appendChild(newchild);
                      }
                    }
                  }
                // })
                }
              }
            }
          }
        }
        //});
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
                      try {
                        const execSync = require('child_process').execSync;
                        const output = execSync(value["cmd"], {shell: true});
                      } catch(Err) {
                        alert("Error when running the command " + value["cmd"] + ": " + Err.message);
                      }
                    }
                    if (value["script"]) {
                      try {
                        eval(value["script"]);
                      } catch(Err) {
                        alert("Error when running the script " + value["script"] + ": " + Err.message);
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
              for (const keyy in inner) {
                if (keyy.startsWith("img_")) {
                  var getting = JSON.parse(inner[keyy]);
                  if (getting["class"] == "operator") {
                    var value = "node-id-" + getting["id"];
                    if (!(onmouseoverdict.hasOwnProperty(value))) {
                      onmouseoverdict[value] = [];
                    }
                    onmouseoverdict[value].push("image-node-id-" + getting["id"] + "_" + keyy);
                  } else {
                    var value = getting["id"] + 1;
                    if (!(onmouseoverdict.hasOwnProperty(value))) {
                      onmouseoverdict[value] = [];
                    }
                    onmouseoverdict[value].push("tensor-image-" + getting["id"] + "_" + keyy);
                  }
                }
              }
            }
            for (const [elem, lista] of Object.entries(onmouseoverdict)) {
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
}

export default Req;