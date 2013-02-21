// Minifier Client for Closure Compiler Service
//

var fs = require("fs"); 
var http = require('http');
var querystring = require('querystring');

console.log("Minifier Client for Closure Compiler Service, Version 1.0");

var arguments = process.argv.splice(2);

if (arguments.length !== 2)
{
	console.log();
	console.log("Usage: node minify.js <input.js> <output.js>");
	console.log();
	return;
}

var sourcePath = arguments[0];
var targetPath = arguments[1];

console.log("  Reading '" + sourcePath + "'...");

fs.exists(sourcePath, function(exists) {
	if (!exists)
	{
		console.log("  '" + sourcePath + "' does not exist.");
	}
	else
	{
		fs.readFile(sourcePath, "utf8", function (error, data) {
			if (error)
			{
				console.log("  Failure reading '" + sourcePath + "'.");
			}
			else
			{
				console.log("  Uploading '" + sourcePath + "'...");

				var postData = querystring.stringify({  
					'compilation_level' : 'SIMPLE_OPTIMIZATIONS',  
					'output_format': 'text',
					'output_info': 'compiled_code',
					'js_code' : data
				});

				var options = {
					method: "POST",
					host: "closure-compiler.appspot.com",
					path: "/compile",
				    headers: {  
						'Content-Type': 'application/x-www-form-urlencoded',  
						'Content-Length': postData.length  
					}
				};

				var request = http.request(options);
				request.on("error", function (error) {
					console.log("  " + error);
				});
				request.on("response", function (response) {
					console.log("  Downloading '" + targetPath + "'...");
					response.setEncoding("utf8");
					var responseData = [];
					response.on("data", function (chunk) {
						responseData.push(chunk);
					});
					response.on("error", function (error) {
						console.log("  " + error);
					});
					response.on("end", function () {
						console.log("  Writing '" + targetPath + "'...");

						fs.writeFile(targetPath, responseData.join(""), function (error) {
							if (error)
							{
								console.log("  " + error);
							}
						});
					});
				});

				request.write(postData);
				request.end();
			}
		});
	}
});
