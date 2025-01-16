import { stringify } from 'querystring';

import { NONAME } from 'dns';

import { execSync } from 'child_process';
import { cp } from 'fs';

class Req {
	// Variable which states if a file was added before.
	file_added = 0;
	// Variable which stores the elements on which 
	// event listeners for images are put.
	event_img = [];
	// Variable which stores the elements on which 
	// event listeners for double click are put.
	event_dbl_click = [];
	constructor() {}
	// Function for cleaning all the cache (information added from the file) in case an error occurs.
	static solve_bugs() {
		var root = document.getElementById("list-modified");
		if (root.length !== 0) {
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
					do {
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
						if (operator.children && operator.children[0] && operator.children[0].children
							 && operator.children[0].children[0]) {
							if (child["hover"] && operator.children[0].children[0].innerHTML !== '') {
								operator.children[0].children[0].innerHTML = '';
							} if (child["style"]
								&& operator.children[0].children[0].hasAttribute("style")) {
								operator.children[0].children[0].removeAttribute("style");
							}
						}
					}
				}
			}
		}
		document.getElementById("list-attributes").innerHTML = '';
		document.getElementById("list-modified").innerHTML = '';
		var body = document.body;
		var children = body.children;
		for (var i = children.length - 1; i >= 0; i--) {
			if (children[i].tagName == 'IMG') {
				body.removeChild(children[i]);
			}
		}
		// Code which removes event listeners for images and doubleclick.
		if (Req.event_img.length !== 0) {
			for (var i = 0; i < Req.event_img.length; i++) {
				(Req.event_img[i]).onmouseover = null;
				(Req.event_img[i]).onmouseout = null;
			}
		}
		if (Req.event_dbl_click.length !== 0) {
			for (var i = 0; i < Req.event_dbl_click.length; i++) {
				Req.event_dbl_click[i].ondblclick = null;
			}
		}
		Req.file_added = 0;
	}
	
	// Function in which file is read, data is processed and buttons, hovers and clicks are added.
	static read_file() {
		let input = document.createElement('input');
		input.type = 'file';
		input.onchange = _ => {
			let files = Array.from(input.files);
			const reader = new FileReader();
			reader.onload = function() {
				try {
				Req.file_added = 1;
				Req.event_img = [];
				Req.event_dbl_click = [];
				const content = reader.result;
				// Variable which stocks the lines from the file.
				var lines = content.split('\n');
				var id, meta, style, specifier;
				// Variable which encapsulates all the data for a graph element to be inserted into html page
				// -> in a list of added attributes.
				var child_meta = 0;
				// Variable which encapsulates the elements' data which needs to be removed manually, 
				// by setting attributes differently -> in a list of modified attributes.
				var another = 0;
				// Variable which stocks the number of buttons added for an element.
				var n_of_buttons = 0;
				// Variable which stocks data for a button and resets every time a new button is created.
				var other_string = JSON.stringify({});
				// Variable which stocks the list of possible keys from a file
				var options_keys = ["operator_onmouseover_image_posx", "operator_onmouseover_image_posy",
					"tensor_onmouseover_image_posx", "tensor_onmouseover_image_posy", "tensor_id",
					"operator_id", "tensor_style", "operator_style", "tensor_ondblclick_script",
					"tensor_ondblclick_command", "operator_ondblclick_script", "operator_ondblclick_command",
					"tensor_onmouseover_text", "operator_onmouseover_text", "tensor_onmouseover_image",
					"operator_onmouseover_image", "tensor_meta_key", "operator_meta_key", "tensor_meta_val",
					"operator_meta_val", "tensor_button", "operator_button", "tensor_button_command",
					"tensor_button_script", "operator_button_command", "operator_button_script",
					"tensor_onmouseover_image_dimx", "tensor_onmouseover_image_dimy",
					"operator_onmouseover_image_dimx", "operator_onmouseover_image_dimy"];
				// Variable which checks if the first line of the file contains an id of tensor or operator.
				var index = -1;
				// Variable which stocks data for a button and resets every time a new
				// hover image is created.
				var img = JSON.stringify({});
				// Variable which stocks the number of hover images created.
				var n_of_images = 0

				for (var line = 0; line < lines.length; line++) {
					// If the line is a comment or if it has no text, it is ignored.
					if (!lines[line].startsWith("#") && lines[line].trim().length !== 0) {
						var line_read = lines[line].split(/:(.+)/);
						// Gets the key of the line.
						var first = line_read[0] == undefined? '' : line_read[0].trim();
						// Checks if the file begins with an id.
						if (index == -1) {
							if (first.toLowerCase() !== "operator_id" &&
							 first.toLowerCase() !== "tensor_id") {
								Req.solve_bugs();
								var text_error = "Invalid model metadata file format!" + 
								" The first line must be operator_id or tensor_id";
								throw text_error;
							}
						}
						index = 0;
						// Checks if the key is not included into the list of valid keys.
						if (!options_keys.includes(first.toLowerCase())) {
							Req.solve_bugs();
							throw "Invalid model metadata file format! You provided an invalid key: " + 
							first.toLowerCase();
						}
						if (first.toLowerCase() == "tensor_id" || first.toLowerCase() == "operator_id") {
							// Block of code which checks if there is unadded information 
							// regarding a new button at the beggining of processing a new graph element.
							if (other_string !== JSON.stringify({})) {
								var objeect = document.getElementById("list-attributes");
								for (var i = 0; i < objeect.children.length; i++) {
									var id_initial = (JSON.parse(objeect.children[i].innerHTML))['id'];
									var id_final = JSON.parse(other_string)['id'];
									if (id_initial == id_final) {
										var inner = JSON.parse(objeect.children[i].innerHTML);
										inner["button_" + n_of_buttons] = other_string;
										objeect.children[i].innerHTML = JSON.stringify(inner);
										other_string = JSON.stringify({});
										n_of_buttons += 1;
										break;
									}
								}
							}
							// Block of code which checks if there is unadded information
							// regarding a new image at the beggining of processing a new graph element.
							if (img !== JSON.stringify({})) {
								var objeect = document.getElementById("list-attributes");
									for (var i = 0; i < objeect.children.length; i++) {
										var id_initial = (JSON.parse(objeect.children[i].innerHTML))['id'];
										var id_final = JSON.parse(img)['id'];
										if (id_initial == id_final) {
											var inner = JSON.parse(objeect.children[i].innerHTML);
											inner["img_" + n_of_images] = img;
											objeect.children[i].innerHTML = JSON.stringify(inner);
											img = JSON.stringify({});
											n_of_images += 1;
											break;
									}
								}
							}
							// Block of code which checks if there is unadded processed information in the
							// lists of new elements at the beginning of processing a new graph element.
							if (child_meta !== 0) {
								document.getElementById("list-attributes").appendChild(child_meta);
								document.getElementById("list-modified").appendChild(another);
								child_meta = 0;
								another = 0;
							}
							// Creating an element which contains the information needed.
							child_meta = document.createElement('div');
							another = document.createElement('div');
							specifier = line_read[0] == undefined? '' : line_read[0].trim();
							if (specifier.toLowerCase() == "tensor_id") {
								child_meta.className = "tensor";
								another.className = "tensor";
							} else {
								child_meta.className = "operator";
								another.className = "operator";
							}
							id = line_read[1] == undefined ? '' : line_read[1].trim();
							if (!(!isNaN(parseFloat(id)) && isFinite(id))) {
								Req.solve_bugs();
								throw "Invalid model metadata file format! Id must be a number";
							}
							child_meta.innerHTML = JSON.stringify({"id": id});
							another.innerHTML = JSON.stringify({"id": id});
							continue;
						}
						// Adding the style info in the element.
						if (first.toLowerCase() == "tensor_style" ||
						 first.toLowerCase() == "operator_style") {
							if ((child_meta.className == "tensor" &&
							first.toLowerCase() == "operator_style") ||
							(child_meta.className == "operator" &&
							first.toLowerCase() == "tensor_style")) {
								Req.solve_bugs();
								throw "Invalid model metadata file format! tensor_style must correspond"
								+ " to tensor and operator_style to operator";
							}
							style = line_read[1] == undefined ? '' : line_read[1].trim();
							var obj = JSON.parse(child_meta.innerHTML);
							obj["style"] = style;
							child_meta.innerHTML = JSON.stringify(obj);
							var obj2 = JSON.parse(another.innerHTML);
							obj2["style"] = style;
							another.innerHTML = JSON.stringify(obj2);
							continue;
						}
						else {
							meta = line_read[1] == undefined ? '' : line_read[1].trim();
							var obj = JSON.parse(child_meta.innerHTML);
							var obj2 = JSON.parse(another.innerHTML);
							// Adding the doubleclick information for the element.
							if (first.toLowerCase() == "tensor_ondblclick_script" ||
							first.toLowerCase() == "tensor_ondblclick_command" ||
							first.toLowerCase() == "operator_ondblclick_script" ||
							first.toLowerCase() == "operator_ondblclick_command") {
								if (((first.toLowerCase() == "tensor_ondblclick_script" ||
								first.toLowerCase() == "tensor_ondblclick_command") &&
								child_meta.className == "operator") ||
								((first.toLowerCase() == "operator_ondblclick_script" ||
								first.toLowerCase() == "operator_ondblclick_command") &&
								child_meta.className == "tensor")) {
									Req.solve_bugs();
									throw "Invalid model metadata file format! tensor_ondblclick.." +
									" must correspond to tensor and operator_ondblclick... to operator";
								}
								obj[first.toLowerCase()] = meta;
							} else if (first.toLowerCase() == "tensor_onmouseover_text" ||
							first.toLowerCase() == "operator_onmouseover_text") {
								if ((first.toLowerCase() == "tensor_onmouseover_text" &&
								child_meta.className == "operator") ||
								(first.toLowerCase() == "operator_onmouseover_text" &&
								child_meta.className == "tensor")) {
									Req.solve_bugs();
									throw "Invalid model metadata file format! tensor_onmouseover_text" + 
									" must correspond to tensor and operator_onmouseover_text to operator";
								}
								obj["hover"] = meta;
								obj2["hover"] = meta;
							}
							// Adding the imahe hovering information for the element.
							else if (first.toLowerCase() == "tensor_onmouseover_image" ||
							first.toLowerCase() == "operator_onmouseover_image") {
								if ((first.toLowerCase() == "tensor_onmouseover_image" &&
								child_meta.className == "operator") ||
								(first.toLowerCase() == "operator_onmouseover_image" &&
								child_meta.className == "tensor")) {
									Req.solve_bugs();
									throw "Invalid model metadata file format! tensor_onmouseover_image" +
									" must correspond to tensor and operator_onmouseover_image to operator";
								}            
								if (img !== JSON.stringify({})) {
									obj["img_" + n_of_images] = img;
									n_of_images += 1;
								}
								var classname = child_meta.className;
								var id = obj["id"];
								img = JSON.stringify({"id": id, "class": classname, "img_link": meta});
								obj2["hover_image"] = meta;
							}
							// Adding the key-value metadata information for the element.
							else if (first.toLowerCase() == "tensor_meta_key" ||
							first.toLowerCase() == "operator_meta_key") {
								if ((first.toLowerCase() == "tensor_meta_key" &&
								child_meta.className == "operator") ||
								(first.toLowerCase() == "operator_meta_key" &&
								child_meta.className == "tensor")) {
									Req.solve_bugs();
									throw "Invalid model metadata file format! tensor_meta_key must " +
									"correspond to tensor and operator_meta_key to operator";
								}
								var k = 1;
								while (lines[line + k].startsWith("#") ||
								lines[line + k].trim().length == 0) {
									k += 1;
								}
								var line_read_second = lines[line + k].split(":");
								var second =
								line_read_second[0] == undefined? '' : line_read_second[0].trim();
								if (second.toLowerCase() == "tensor_meta_val" ||
								second.toLowerCase() == "operator_meta_val") {
									if ((first.toLowerCase() == "tensor_meta_val" &&
									child_meta.className == "operator") ||
									(first.toLowerCase() == "operator_meta_val" &&
									child_meta.className == "tensor")) {
										Req.solve_bugs();
										throw "Invalid model metadata file format! tensor_meta_val "
										+ "must correspond to tensor and operator_meta_val to operator";
									}
									var value = line_read_second[1].trim();
									var key = line_read[1].trim();
									obj[key] = value;
									obj2["metadata"] = "metadata";
								} else {
									Req.solve_bugs();
									throw "Invalid model metadata file format! meta_val must" +
									" be after meta_key";
								}
							}
							else if (first.toLowerCase() == "tensor_meta_val" ||
							first.toLowerCase() == "operator_meta_val" ) {
								var k = -1;
								while (lines[line + k].startsWith("#") ||
								lines[line + k].trim().length == 0) {
									k -= 1;
								}
								var line_read_second = lines[line + k].split(":");
								var second =
								line_read_second[0] == undefined? '' : line_read_second[0].trim();
								if ((second.toLowerCase() == "tensor_meta_key" &&
								first.toLowerCase() == "tensor_meta_val") ||
								(second.toLowerCase() == "operator_meta_key" &&
								first.toLowerCase() == "operator_meta_val")) {
									continue;
								} else if ((second.toLowerCase() == "tensor_meta_key" &&
								first.toLowerCase() == "operator_meta_val") ||
								(second.toLowerCase() == "operator_meta_key" &&
								first.toLowerCase() == "tensor_meta_val")) {
									Req.solve_bugs();
									throw "Invalid model metadata file format! both key and value must " +
									"be part of tensor or operator";
								} else {
									Req.solve_bugs();
									throw "Invalid model metadata file format! meta_val must "
									+ "be after meta_key";
								}
							}
							else {
								// Adding the button information for the element.
								if (first.toLowerCase() == "tensor_button" ||
								first.toLowerCase() == "operator_button") {
									if ((child_meta.className == "operator" &&
									first.toLowerCase() == "tensor_button") ||
									(child_meta.className == "tensor" &&
									first.toLowerCase() == "operator_button")) {
										Req.solve_bugs();
										throw "Invalid model metadata file format! tensor_button must " +
										"correspond to tensor and operator_button to operator";
									}
									var k = 1;
									while (lines[line + k].startsWith("#") ||
									lines[line + k].trim().length == 0) {
										k += 1;
									}
									var line_read_second = lines[line + k].split(":");
									var elem = line_read_second[0];
									var second = elem == undefined? '' : elem.trim();
									if ((first.toLowerCase() == "tensor_button" &&
									second.toLowerCase() == "tensor_button") ||
									(first.toLowerCase() == "operator_button" &&
									second.toLowerCase() == "operator_button")) {
										Req.solve_bugs();
										throw "Invalid model metadata file format! " + 
										"You can't have empty buttons";
									}
									if (other_string !== JSON.stringify({})) {
										obj["button_" + n_of_buttons] = other_string;
										n_of_buttons += 1;
									}
									var id = obj["id"];
									var c_name = child_meta.className;
									var other_s = line_read[1] == undefined ? '' : line_read[1].trim();
									JSON.stringify({"id": id, "class": c_name, "button_name": other_s});
									obj2["buttons"] = "buttons";
								} else {
									if (first.toLowerCase() == "tensor_button_command" ||
									first.toLowerCase() == "tensor_button_script" ||
									first.toLowerCase() == "operator_button_command" ||
									first.toLowerCase() == "operator_button_script") {
										if ((child_meta.className == "tensor" &&
											(first.toLowerCase() == "operator_button_command" ||
											first.toLowerCase() == "operator_button_script")) ||
											(child_meta.className == "operator" &&
												(first.toLowerCase() == "tensor_button_command" ||
												first.toLowerCase() == "tensor_button_script"))) {
											Req.solve_bugs();
											throw "Invalid model metadata file format! tensor_button.." +
											" must correspond to tensor and operator_button.. to operator";
										}
										var k = -1;
										while (lines[line + k].startsWith("#") ||
										lines[line + k].trim().length == 0) {
											k -= 1;
										}
										var line_read_second = lines[line + k].split(":");
										var second =
										line_read_second[0] == undefined? '' : line_read_second[0].trim();
										if ((first.toLowerCase() == "tensor_button_command" &&
										second.toLowerCase() == "tensor_button_command") ||
										(first.toLowerCase() == "tensor_button_script" &&
										second.toLowerCase() == "tensor_button_script") ||
										(first.toLowerCase() == "operator_button_command" &&
										second.toLowerCase() == "operator_button_command") ||
										(first.toLowerCase() == "operator_button_script" &&
										second.toLowerCase() == "operator_button_script")) {
											Req.solve_bugs();
											throw "A button can't execute two scripts or " +
											"two commands at the same time";
										}
										if (first.toLowerCase() == "tensor_button_command") {
											if (second.toLowerCase() !== "tensor_button_script" &&
											second.toLowerCase() !== "tensor_button") {
												Req.solve_bugs();
												throw "The format of the data is wrong: the construction" +
												" of the button should be tensor_button, " +
												"tensor_button_command, tensor_button_script or " +
												"tensor_button, tensor_button_script, tensor_button_command";
											}
										}
										if (first.toLowerCase() == "tensor_button_script") {
											if (second.toLowerCase() !== "tensor_button_command" &&
											second.toLowerCase() !== "tensor_button") {
												Req.solve_bugs();
												throw "The format of the data is wrong: the construction" +
												" of the button should be tensor_button, " +
												"tensor_button_command, tensor_button_script or " +
												"tensor_button, tensor_button_script, tensor_button_command";
											}
										}
										if (first.toLowerCase() == "operator_button_command") {
											if (second.toLowerCase() !== "operator_button_script" &&
											second.toLowerCase() !== "operator_button") {
												Req.solve_bugs();
												throw "The format of the data is wrong: the construction" +
												" of the button should be operator_button, " +
												"operator_button_command, operator_button_script or " +
												"operator_button, operator_button_script," +
												" operator_button_command";
											}
										}
										if (first.toLowerCase() == "operator_button_command") {
											if (second.toLowerCase() !== "operator_button_script" &&
											second.toLowerCase() !== "operator_button") {
												Req.solve_bugs();
												throw "The format of the data is wrong: the construction" +
												" of the button should be operator_button, " +
												"operator_button_command, operator_button_script or " +
												"operator_button, operator_button_script, " +
												"operator_button_command";
											}
										}
										var stri = JSON.parse(other_string);
										stri[first.toLowerCase()] =
										line_read[1] == undefined ? '' : line_read[1].trim();
										other_string = JSON.stringify(stri);
									} 
									else {
										// Adding the image information for the element.
										if (first.toLowerCase() == "tensor_onmouseover_image_dimx" ||
										first.toLowerCase() == "tensor_onmouseover_image_dimy" ||
										first.toLowerCase() == "operator_onmouseover_image_dimx" ||
										first.toLowerCase() == "operator_onmouseover_image_dimy" ||
										first.toLowerCase() == "tensor_onmouseover_image_posx" ||
										first.toLowerCase() == "tensor_onmouseover_image_posy" ||
										first.toLowerCase() == "operator_onmouseover_image_posx" ||
										first.toLowerCase() == "operator_onmouseover_image_posy") {
											if ((child_meta.className == "tensor" &&
											first.toLowerCase().startsWith("operator_")) ||
											(child_meta.className == "operator" &&
												first.toLowerCase().startsWith("tensor_"))) {
												Req.solve_bugs();
												throw "Invalid model metadata file format! "+
												"tensor_onmouseover_image.. must correspond to tensor and" +
												" operator_onmouseover_image.. to operator";
											}
											var k = -1;
											while (lines[line + k].startsWith("#") ||
											lines[line + k].trim().length == 0) {
												k -= 1;
											}
											var line_read_second = lines[line + k].split(":");
											var second = line_read_second[0] ==
											undefined? '' : line_read_second[0].trim();
											if (first.toLowerCase() == second.toLowerCase()) {
												Req.solve_bugs();
												throw "A button can't execute two x dimesions or two y " +
												"dimensions at the same time";
											}
											var strin = JSON.parse(img);
											strin[first.toLowerCase()] =
											line_read[1] == undefined ? '' : line_read[1].trim();
											img = JSON.stringify(strin);
										} else {
											// If there is unadded information for buttons and images,
											// this will be added.
											if (other_string !== JSON.stringify({})) {
												obj["button_" + n_of_buttons] = other_string;
											}
											if (img !== JSON.stringify({})) {
												obj["img_" + n_of_images] = img;
											}
											other_string = JSON.stringify({});
											img = JSON.stringify({});
											Req.solve_bugs();
											throw "You provided an invalid key: " + first.toLowerCase();
										}
									}
								}
							}
							child_meta.innerHTML = JSON.stringify(obj);
							another.innerHTML = JSON.stringify(obj2);
							continue;
						} 
					}
				}
				// If there is unprocessed information, add it in the list.
				if (child_meta !== 0) {
					document.getElementById("list-attributes").appendChild(child_meta);
					document.getElementById("list-modified").appendChild(another);
				}
				// If there is unprocessed button or image information,
				// process it.
				if (other_string !== JSON.stringify({})) {
				var objeect = document.getElementById("list-attributes");
					for (var i = 0; i < objeect.children.length; i++) {
						if ((JSON.parse(objeect.children[i].innerHTML))['id'] ==
						JSON.parse(other_string)['id']) {
							var inner = JSON.parse(objeect.children[i].innerHTML);
							inner["button_" + n_of_buttons] = other_string;
							objeect.children[i].innerHTML = JSON.stringify(inner);
							other_string = JSON.stringify({});
							n_of_buttons += 1;
							break;
						}
					}
				}
				if (img !== JSON.stringify({})) {
					var objeect = document.getElementById("list-attributes");
					for (var i = 0; i < objeect.children.length; i++) {
						if ((JSON.parse(objeect.children[i].innerHTML))['id'] ==
						JSON.parse(img)['id']) {
							var inner = JSON.parse(objeect.children[i].innerHTML);
							inner["img_" + n_of_images] = img;
							objeect.children[i].innerHTML = JSON.stringify(inner);
							img = JSON.stringify({});
							n_of_images += 1;
							break;
						}
					}
				}
				// Block of code which validates the changes : adds the style,
				// the hovering and the buttons.
				if (document.getElementById("list-attributes").children) {
					var list_attr = document.getElementById("list-attributes");
					var list_modif = document.getElementById("list-modified");
					var parent_t = document.getElementById("edge-paths");
					var parent_n = document.getElementById("nodes");
					for (var i = 0; i < list_attr.children.length; i++) {
						if (list_attr.children[i].className == "tensor") {
							var results = [];
							var len = parent_t.children.length;
							for (var idx = 0; idx <= len; idx++) {
								results.push([idx, parent_t.children[idx]]);
							}
							var flag = 0;
							for (var idx = 0; idx < len; idx++) {
								var obj = JSON.parse(list_attr.children[i].innerHTML);
								var obj2 = JSON.parse(list_modif.children[i].innerHTML);
								if (results[idx][1].id.split("\n")[1] == obj['id']) {
									obj["new_id"] = idx;
									obj["tensorname"] = results[idx][1].id;
									obj2["new_id"] = idx;
									obj2["tensorname"] = results[idx][1].id;
									list_attr.children[i].innerHTML = JSON.stringify(obj);
									if (obj["style"]) {
										parent_t.children[idx].style.stroke = obj["style"];
									}
									flag = 1;
									if (obj["hover"]) {
										parent_t.children[idx + 1].innerHTML = '<title>' +
										obj["hover"].match(/.{1,20}/g).join("\n") + '</title>';
									}
									var keys = Object.keys(obj);
									// Block of code whick adds image item to be displayed.
									for (var j = 0; j < keys.length; j++) {
										if (keys[j].startsWith("img_")) {
											var obje = JSON.parse(obj[keys[j]]);
											var image = document.createElement('img');
											image.src = obje["img_link"];
											image.style.width =
											obje.hasOwnProperty("tensor_onmouseover_image_dimx") ?
											obje["tensor_onmouseover_image_dimx"] : "auto";
											image.style.height =
											obje.hasOwnProperty("tensor_onmouseover_image_dimy") ?
											obje["tensor_onmouseover_image_dimy"] : "auto";
											image.style.display = 'none';
											image.style.position = 'absolute';
											image.style.top =
											obje.hasOwnProperty("tensor_onmouseover_image_posx") ?
											obje["tensor_onmouseover_image_posx"] : "auto";
											image.style.left =
											obje.hasOwnProperty("tensor_onmouseover_image_posy") ?
											obje["tensor_onmouseover_image_posy"] : "auto";
											image.id = "tensor-image-" + idx + "_" + keys[j];
											image.className = "put-image-on-hover";
											document.body.appendChild(image);
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
						if (list_attr.children[i].className == "operator") {
							var obj = JSON.parse(list_attr.children[i].innerHTML);
							var obj2 = JSON.parse(list_modif.children[i].innerHTML);
							var id = obj["id"];
							var idx = 1;
							var counter = 0;
							var flag = 0;
							do {
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
							var new_obj = JSON.parse(list_attr.children[i].innerHTML);
							new_obj["new_id"] = counter;
							obj2["new_id"] = counter;
							list_attr.children[i].innerHTML = JSON.stringify(new_obj);
							list_modif.children[i].innerHTML = JSON.stringify(obj2);
							var operator = document.getElementById("node-id-" + counter);
							if (!operator) {
								Req.solve_bugs();
								throw "Index of operator not found :" + obj["id"];
							}
							if (obj["hover"]) {
								operator.children[0].children[0].innerHTML = '<title>' +
								obj["hover"].match(/.{1,20}/g).join("\n") + '</title>';
							}
							var keys = Object.keys(obj);
							for (var j = 0; j < keys.length; j++) {
								if (keys[j].startsWith("img_")) {
									var obje = JSON.parse(obj[keys[j]]);
									var image = document.createElement('img');
									image.src = obje["img_link"];
									image.style.width =
									obje.hasOwnProperty("operator_onmouseover_image_dimx") ?
									obje["operator_onmouseover_image_dimx"] : "auto";
									image.style.height =
									obje.hasOwnProperty("operator_onmouseover_image_dimy") ?
									obje["operator_onmouseover_image_dimy"] : "auto";
									image.style.display = 'none';
									image.style.position = 'absolute';
									image.style.top =
									obje.hasOwnProperty("operator_onmouseover_image_posx") ?
									obje["operator_onmouseover_image_posx"] : "auto";
									image.style.left =
									obje.hasOwnProperty("operator_onmouseover_image_posy") ?
									obje["operator_onmouseover_image_posy"] : "auto";
									image.id = "image-node-id-" + counter + "_" + keys[j];
									image.className = "put-image-on-hover";
									document.body.appendChild(image);
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
	// Function which manages the doubleclick feature.
	static doubleclick() {
		if (document.getElementById("list-attributes").children) {
			var parent_t = document.getElementById("edge-paths");
			var parent_n = document.getElementById("nodes");
			var list_attr = document.getElementById("list-attributes").children;
			for (var i = 0; i < list_attr.length; i++) {
				if (list_attr[i].className === "tensor") {
					var op = JSON.parse(list_attr[i].innerHTML);
					var id = op["tensorname"];
					list_attr[i].id = op["new_id"] + 1;
				} else {
					var op = JSON.parse(list_attr[i].innerHTML);
					var id = "node-id-" + op["new_id"];
					list_attr[i].id = id;
				}
			}
			for (var i = 0; i < list_attr.length; i++) {
				if (list_attr[i].className == "tensor") {
					var variable = list_attr[i].id;
					(parent_t.children[variable]).addEventListener("dblclick", function(e) {
						var idx = 0;
						var op;
						for (var i = 0; i < parent_t.children.length; i++) {
							if (parent_t.children[i].isEqualNode(e.srcElement)) {
								idx = i;
								break;
							}
						}
						for (var i = 0; i < list_attr.length; i++) {
							if (list_attr[i].id == idx) {
								op = JSON.parse(list_attr[i].innerHTML);
								break;
							}
						}
						if (op["tensor_ondblclick_command"]) {
							try {
								const execSync = require('child_process').execSync;
								const output = execSync(op["tensor_ondblclick_command"], {shell: true});
							} catch(Err) {
								alert("Error when running the command " +
								op["tensor_ondblclick_command"] + ": " + Err.message);
							}
						}
						if (op["tensor_ondblclick_script"]) {
							try {
								eval(op["tensor_ondblclick_script"]);
							} catch(Err) {
								var result = op["tensor_ondblclick_script"];
								alert("Error when running the script " + result + ": " + Err.message);
							}
						}
					})
					Req.event_dbl_click.push(parent_t.children[variable]);
				} else {
					var variable = list_attr[i].id;
					(document.getElementById(variable)).addEventListener("dblclick", function(g) {
						var idx = 0;
						var op;
						for (var i = 0; i < parent_n.children.length; i++) {
							if ((parent_n.children[i].children[0].children[0]).isEqualNode(g.srcElement) ||
							(parent_n.children[i].children[0].children[1]).isEqualNode(g.srcElement)) {
								idx = parent_n.children[i].id;
								break;
							}
						}
						for (var i = 0; i < list_attr.length; i++) {
							if (list_attr[i].id == idx) {
								op = JSON.parse(list_attr[i].innerHTML);
								break;
							}
						}
						if (op["operator_ondblclick_command"]) {
							try {
								const execSync = require('child_process').execSync;
								const output = execSync(op["operator_ondblclick_command"], {shell: true});
							} catch(Err) {
								var result = op["operator_ondblclick_command"];
								alert("Error when running the command " + result + ": " + Err.message);
							}
						}
						if (op["operator_ondblclick_script"]) {
							try {
								eval(op["operator_ondblclick_script"]);
							} catch(Err) {
								var result = op["operator_ondblclick_script"];
								alert("Error when running the script " + result + ": " + Err.message);
							}
						}
					})
					Req.event_dbl_click.push(document.getElementById(variable));
				}
			}
		}
	}
// Function which deals with displaying all the other information.
	static metadata() {
		if (document.getElementById("list-attributes").children) {
			var list_attr = document.getElementById("list-attributes");
			var parent_t = document.getElementById("edge-paths");
			var parent_n = document.getElementById("nodes");
			for (var i = 0; i < list_attr.children.length; i++) {
				var element_searched = list_attr.children[i];
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
							var sidebar_obj = document.getElementById("sidebar-content").children[0];
							var counter = 0;
							for (var j = 0; j< sidebar_obj.children.length; j++) {
								var childd = sidebar_obj.children[j];
								if (childd.textContent !== "Metadata" && childd.innerText !== "Metadata" &&
								childd.innerHTML !== "Metadata") {
									counter += 1;
								}
							}
							if (counter == sidebar_obj.children.length) {
								if ((element_searched.className == "operator" &&
								where.className['baseVal'] == "node graph-node select") ||
								(element_searched.className == "tensor" &&
								where.className['baseVal'] == "edge-path select")) {
									var keys = Object.keys(inner);
									const child_meta = document.createElement('div');
									child_meta.className = "sidebar-header";
									child_meta.innerText = "Metadata";
									sidebar_obj.appendChild(child_meta);
									for (var i = 0; i < keys.length; i++) {
										if (keys[i] !== "id" && keys[i] !== "style" &&
										keys[i] !== "new_id" && keys[i] !== "tensorname" &&
										keys[i].slice(0, 7) !== "button_" &&
										keys[i] !== "tensor_ondblclick_script" &&
										keys[i] !== "operator_ondblclick_script" &&
										keys[i] !== "tensor_ondblclick_command" &&
										keys[i] !== "operator_ondblclick_command" &&
										keys[i] !== "hover" && keys[i] !== "hover_image" &&
										keys[i].slice(0, 4) !== "img_") {
											var new_child = document.createElement('div');
											new_child.className = "sidebar-item";
											var new_child_2 = document.createElement('div');
											new_child_2.className = "sidebar-item-name";
											var inputc = document.createElement('input');
											inputc.type="text";
											inputc.value=keys[i];
											inputc.title=keys[i];
											inputc.readonly="true";

											var new_child_3 = document.createElement('div');
											new_child_3.className = "sidebar-item-value-list";

											var new_child_4 = document.createElement('div');
											new_child_4.className = "sidebar-item-value";

											var new_child_5 = document.createElement('div');
											new_child_5.className = "sidebar-item-value-expander";
											new_child_5.innerText = '+';

											var new_child_6 = document.createElement('div');
											new_child_6.className = "sidebar-item-value-line";
											new_child_6.style="cursor: pointer;";

											var new_child_7 = document.createElement('span');
											new_child_7.className = "sidebar-item-value-line-content";
											new_child_7.innerHTML = `<b>` + inner[keys[i]] + `</b>`;
											
											new_child_6.append(new_child_7);
											new_child_4.append(new_child_5);
											new_child_4.append(new_child_6);
											new_child_3.append(new_child_4);
											new_child_2.append(inputc);
											new_child.appendChild(new_child_2);
											new_child.appendChild(new_child_3);
											sidebar_obj.appendChild(new_child);
										}
									}
									var repeated = 0;
									for (var i = 0; i < keys.length; i++) {
										if (keys[i].slice(0, 7) == "button_") {
											if (repeated == 0) {
												const child_meta = document.createElement('div');
												child_meta.className = "sidebar-header";
												child_meta.innerText = "Actions";
												sidebar_obj.appendChild(child_meta);
												repeated = 1;
											}
											var string_new = JSON.parse(inner[keys[i]]);
											if (inner["id"] == string_new["id"]) {
												var new_child = document.createElement('button');
												new_child.type = "button";
												new_child.textContent = string_new["button_name"];
												new_child.style = "display: block; margin: 10px;" +  
												"background-color:#99c2ff; color:black; border-radius: 8px;";
												if (string_new["class"] == "tensor") {
													var cmd = string_new["tensor_button_command"];
													var scr = string_new["tensor_button_script"];
													var value;
													if (cmd && str) {
														value = JSON.stringify({"cmd": cmd, "script": scr});
													} else {
														if (cmd) {
															value += JSON.stringify({"cmd": cmd});
														} else if (scr) {
															value += JSON.stringify({"script": scr});
														} else {
															value += JSON.stringify({});
														}
													}
													new_child.value = value;
												}
												if (string_new["class"] == "operator") {
													var cmd = string_new["operator_button_command"];
													var scr = string_new["operator_button_script"];
													var value;
													if (cmd && scr) {
														value = JSON.stringify({"cmd": cmd, "script": scr});
													} else {
														if (cmd) {
															value += JSON.stringify({"cmd": cmd});
														} else if (scr) {
															value += JSON.stringify({"script": scr});
														} else {
															value += JSON.stringify({});
														}
													}
													new_child.value = value;
												}
												sidebar_obj.appendChild(new_child);
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
		// Block of code which adds execution of scripts or commands when
		// pressing a button.
		if (document.getElementById("list-attributes").children &&
		document.getElementById("sidebar-content") &&
		document.getElementById("sidebar-content").children[0] &&
		document.getElementById("sidebar-content").children[0].children) {
			var sidebar_obj = document.getElementById("sidebar-content").children[0];
			for (var i = 0; i < sidebar_obj.children.length; i++) {
				if (sidebar_obj.children[i].type == "button") {
					if (sidebar_obj.children[i].getAttribute('listener') !== 'true') {
						sidebar_obj.children[i].setAttribute('listener', 'true');
						sidebar_obj.children[i].addEventListener('click', function(h) {
							var value = JSON.parse(h.target.value);
							if (value["cmd"]) {
								try {
									const execSync = require('child_process').execSync;
									const output = execSync(value["cmd"], {shell: true});
								} catch(Err) {
									var cmd = value["cmd"];
									alert("Error when running the command " + cmd + ": " + Err.message);
								}
							}
							if (value["script"]) {
								try {
									eval(value["script"]);
								} catch(Err) {
									var scr = value["script"];
									alert("Error when running the script " + scr + ": " + Err.message);
								}
							}
						});
					}
				}
			}
		}
		// Block of code which displays the images on hover.
		if (document.getElementById("graph") && document.getElementById("origin") &&
		document.getElementById("nodes") && document.getElementById("edge-paths") &&
		document.getElementById("list-attributes").innerHTML.length !== 0) {
			var parent_n = document.getElementById("nodes");
			var parent_t = document.getElementById("edge-paths");
			var list_attr = document.getElementById("list-attributes");
			var onmouseover_dict = {};
			for (var i = 0; i < list_attr.children.length; i++) {
				var child = list_attr.children[i];
				var inner = JSON.parse(child.innerHTML);
				var item;
				var elem;
				for (const keyy in inner) {
					if (keyy.startsWith("img_")) {
						var getting = JSON.parse(inner[keyy]);
						if (getting["class"] == "operator") {
							var value = "node-id-" + inner["new_id"];
							if (!(onmouseover_dict.hasOwnProperty(value))) {
								onmouseover_dict[value] = [];
							}
							onmouseover_dict[value].push("image-node-id-" + inner["new_id"] + "_" + keyy);
						} else {
							var value = inner["new_id"] + 1;
							if (!(onmouseover_dict.hasOwnProperty(value))) {
								onmouseover_dict[value] = [];
							}
							onmouseover_dict[value].push("tensor-image-" + inner["new_id"] + "_" + keyy);
						}
					}
				}
			}
			for (const [elem, list_a] of Object.entries(onmouseover_dict)) {
				if (elem.startsWith("node")) {
					document.getElementById(elem).onmouseover = function() {
						for (const item of list_a) {
							document.getElementById(item).style.display = "block";
						}
					}
					document.getElementById(elem).onmouseout = function() {
						for (const item of list_a) {
							document.getElementById(item).style.display = "none";
						}
					}
					Req.event_img.push(document.getElementById(elem));
				} else {
					parent_t.children[elem].onmouseover = function() {
						for (const item of list_a) {
							document.getElementById(item).style.display = "block";
						}
					}
					parent_t.children[elem].onmouseout = function() {
						for (const item of list_a) {
							document.getElementById(item).style.display = "none";
						}
					}
					Req.event_img.push(parent_t.children[elem]);
				}
			}
		}
	}
}

export default Req;
