class Scr {
  constructor() {}
  static async _request2(url) {
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
                } else {
                  var decoded = atob(textrec);
                  // var base64img = "data:image/jpg;base64," + decoded;
                  // var imgel = document.createElement("img");
                  // imgel.src = decoded;
                  console.log(decoded);
                  //(document.getElementById("sidebar-content").children[0]).append(imgel);
                }
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

  static addbutton() {
    if (document.getElementById("graph")) {
      if (document.getElementById("origin")) {
        if (document.getElementById("nodes")) {
          var child = document.getElementById("nodes").children[0];
          if (child.children[0].getAttribute("listener") !== 'true') {
            child.children[0].setAttribute('listener', 'true');
            child.children[0].addEventListener("click", function() {
              console.log("am apasat pe input");
              var sidebarobj = document.getElementById("sidebar-content").children[0];
              var flag = 0;
              for (var i = 0; i < sidebarobj.children.length; i++) {
                if (sidebarobj.children[i].type == "button") {
                  flag = 1;
                }
              }
              if (flag == 0) {
                console.log("flag ul este 0");
                var newchild = document.createElement('button');
                newchild.type = "button";
                newchild.innerText = "Button for python script";
                newchild.webkitdirectory = 'true';
                if (newchild.getAttribute('listener') !== 'true') {
                  newchild.setAttribute('listener', 'true');
                  newchild.addEventListener('click', function() {
                      var path = prompt("Please provide the path");
                      var result = Scr._request2("/pathis_" + path);
                      console.log(result);
                  });
                }
                sidebarobj.appendChild(newchild);
              }
            });
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