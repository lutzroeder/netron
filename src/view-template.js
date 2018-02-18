/*jshint esversion: 6 */

var inputTemplate = `
<style type='text/css'>
.inputs { font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; font-size: 12px; line-height: 1.5; margin: 0; }
.input { margin-bottom: 20px; }
.input b { font-weight: 600; }
.input h1 { font-weight: 600; font-size: 14px; line-height: 1.25; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 0; margin-bottom: 16px; }
.input code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); padding: 0.2em 0.4em; margin: 2px 0 2px 0; border-radius: 3px; }
.input pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 11px; padding: 8px 12px 8px 12px; overflow: auto; line-height: 1.45; margin: 2px 0 2px 0; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; }
.connection { margin-top: 8px; background-color: rgba(27, 31, 35, 0.05); border-radius: 8px; border: 1px solid rgba(27, 31, 35, 0.05); }
.connection-field { font-size: 10px; padding: 4px 8px 4px 8px; }
.connection-border { border-top: 1px solid rgba(27, 31, 35, 0.05); }
.connection code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(0, 0, 0, 0); padding: 0; margin: 0; border: 0; }
.connection pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(0, 0, 0, 0); margin: 0; padding: 4px 8px 4px 8px; border: 0; }
</style>
<div class='inputs'>
<div class='input'>
<b>{{{name}}}{{#if type}}: {{/if}}</b>{{#if type}}<code><b>{{{type}}}</b></code>{{/if}}<br>
{{#connections}}
<div class='connection'>
<div class='connection-field'>
connection: <b>{{{id}}}</b>
{{#if initializer}}
{{#if initializer.title}}
<div style='float: right;'>{{initializer.title}}</div>
{{/if}}
{{/if}}
</div>
{{#if initializer.description}}
<div class='connection-border' />
<div class='connection-field' />
{{{initializer.description}}}
</div>
{{/if}}
{{#if type}}
<div class='connection-border' />
<div class='connection-field'>
type: <code><b>{{{type}}}</b></code>
</div>
{{/if}}
{{#if initializer.quantization}}
<div class='connection-border' />
<div class='connection-field'>
quantization: <code>{{{initializer.quantization}}}</code>
</div>
{{/if}}
{{#if initializer.value}}
<div class='connection-border' />
<pre>{{{initializer.value}}}</pre>
{{/if}}
</div>
{{/connections}}
</div>
</div>
`;

var operatorTemplate = `
<style type='text/css'>
.documentation { font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; font-size: 12px; line-height: 1.5; margin: 0; }
.documentation h1 { font-weight: 600; font-size: 14px; line-height: 1.25; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 0; margin-bottom: 16px; }
.documentation h2 { font-weight: 600; font-size: 12px; line-height: 1.25; margin-bottom: 16px; border-bottom: 1px solid #eaecef; }
.documentation h3 { font-weight: 600; font-size: 12px; line-height: 1.25; }
.documentation p { }
.documentation code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); padding: 0.2em 0.4em; margin: 0; border-radius: 3px }
.documentation pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 11px; padding: 16px; overflow: auto; line-height: 1.45; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; }
.documentation pre code { font-size: 11px; padding: 16px; line-height: 1.45; background-color: transparent; padding: 0; border-radius: 0; }
.documentation tt { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-weight: 600; font-size: 85%; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; padding: 0.2em 0.4em; margin: 0; }
.documentation dl dt { font-size: 12px; font-weight: 600; padding: 0; margin-top: 16px; }
.documentation dd { padding: 0 16px; margin-left: 0; margin-bottom: 16px; }
</style>

<div id='documentation' class='documentation'>

<h1>{{{name}}}</h1>
{{#if summary}}
<p>{{{summary}}}</p>
{{/if}}
{{#if description}}
<p>{{{description}}}</p>
{{/if}}

{{#if attributes}}
<h2>Attributes</h2>
<dl>
{{#attributes}}
<dt>{{{name}}}{{#if type}}: <tt>{{{type}}}</tt>{{/if}}</dt>
<dd>{{{description}}}</dd>
{{/attributes}}
</dl>
{{/if}}

{{#if inputs}}
<h2>Inputs{{#if inputs_range}} ({{{inputs_range}}}){{/if}}</h2>
<dl>
{{/if}}
{{#inputs}}
<dt>{{{name}}}{{#if type}}: <tt>{{{type}}}</tt>{{/if}} {{#if option}}({{{option}}}){{/if}}</dt>
<dd>{{{description}}}</dd>
{{/inputs}}
</dl>

{{#if outputs.length}}
<h2>Outputs{{#if outputs_range}} ({{{outputs_range}}}){{/if}}</h2>
<dl>
{{/if}}
{{#outputs}}
<dt>{{{name}}}{{#if type}}: <tt>{{{type}}}</tt>{{/if}} {{#if option}}({{{option}}}){{/if}}</dt>
<dd>{{{description}}}</dd>
{{/outputs}}
</dl>

{{#if type_constraints}}
<h2>Type Constraints</h2>
<dl>
{{#type_constraints}}
<dt>{{{type_param_str}}}: {{#allowed_type_strs}}<tt>{{this}}</tt>{{#unless @last}}, {{/unless}}{{/allowed_type_strs}}</dt>
<dd>{{{description}}}</dd>
{{/type_constraints}}
</dl>
{{/if}}

{{#if snippets}}
<h2>Examples</h2>
{{/if}}
{{#snippets}}
<h3>{{{summary}}}</h3>
<pre>{{{code}}}</pre>
{{/snippets}}

{{#if domain}}{{#if since_version}}{{#if support_level}}
<h2>Support</h2>
<dl>
In domain <tt>{{{domain}}}</tt> since version <tt>{{{since_version}}}</tt> at support level <tt>{{{support_level}}}</tt>.
</dl>
{{/if}}{{/if}}{{/if}}

</div>
`;

var summaryTemplate = `
<style type='text/css'>
.summary { font-family: 'Open Sans', --apple-system, "Helvetica Neue", Helvetica, Arial, sans-serf; font-size: 12px; line-height: 1.5; overflow: hidden; margin: auto; }
.summary h1 { font-family: 'Open Sans', --apple-system, "Helvetica Neue", Helvetica, Arial, sans-serf; font-weight: 600; font-size: 12px; margin: 0; color: #666; letter-spacing: 0.5px; padding: 10px 0px 0px 0px; margin: 20px 0px 0px 0px; -webkit-user-select: none; -moz-user-select: none; user-select: none; }
.summary .section { margin-top: 10px; margin-bottom: 10px; padding: 10px; overflow-y: auto; position: relative; border: 1px solid none; }
.summary .section table code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); padding: 0.2em 0.4em; margin: 0; border-radius: 3px; background-color: #d6d6d6; }
.summary .border { border-radius: 10px; border: 1px solid #ccc; }
.summary .section table { color: #777; float: left; font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; font-size: 12px; border-spacing: 0; line-height: 150%; }
.summary .section table td:first-child { font-weight: 600; width: 80px; vertical-align: top; user-select: none; -webkit-user-select: none; -moz-user-select: none; }
.summary .section table td { user-select: text; -webkit-user-select: text; -moz-user-select: text; }
</style>
<div class='summary'>
<h1>MODEL PROPERTIES</h1>
<div class='border section'>
<table>
{{#properties}}
<tr><td>{{name}}</td><td>{{value}}</td></tr>
{{/properties}}
</table>
</div>
<h1>GRAPHS</h1>
{{#graphs}}
<div class='border section'>
<table>
{{#if name}}
<tr><td>Name</td><td>{{name}}</td></tr>
{{/if}}
{{#if version}}
<tr><td>Version</td><td>{{version}}</td></tr>
{{/if}}
{{#if type}}
<tr><td>Type</td><td>{{type}}</td></tr>
{{/if}}
{{#if tags}}
<tr><td>Tags</td><td>{{tags}}</td></tr>
{{/if}}
<tr>
{{#if inputs}}
<td>Inputs</td>
<td>
{{#inputs}}
<code>{{type}}</code> : {{name}}<br>
{{/inputs}}
</td>
</tr>
{{/if}}
{{#if outputs}}
<tr>
<td>Outputs</td>
<td>
{{#outputs}}
<code>{{type}}</code> : {{name}}<br>
{{/outputs}}
</td>
</tr>
{{/if}}
{{#if description}}
<tr>
<td>Description</td>
<td>{{description}}</td>
</tr>
{{/if}}
</table>
<button style='float: right; width: 80px;' onclick='javascript:updateActiveGraph("{{{name}}}");'>View</button>
</div>
{{/graphs}}
</div>
`;

var nodeTemplate = `
<style type='text/css'>
.node-summary h1 { font-weight: 600; font-size: 14px; line-height: 1.25; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 0; margin-bottom: 16px; }
.node-summary h2 { font-weight: 600; font-size: 12px; line-height: 1.25; margin-bottom: 16px; border-bottom: 1px solid #eaecef; }
.node-summary h3 { font-weight: 600; font-size: 12px; line-height: 1.25; }
.node-summary .documentation-button { display: inline-block; text-align: center; vertical-align: middle; font-weight: 600; width: 12px; height: 12px; font-size: 10px; line-height: 12px; border-radius: 50%; transform: translateY(-1px); padding: 1px; color: #888; background: transparent; border: 1px solid #aaa; text-decoration: none; }
.node-summary .documentation-button:hover { color: #f6f6f6; background: #aaa; border-color: #aaa; text-decoration: none; }
.node-summary .node-group { font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; font-size: 12px; line-height: 1.5; margin: 0; }
.node-summary .node-group .node-item { margin-bottom: 20px; }
.node-summary .node-group .node-item b { font-weight: 600; }
.node-summary .node-group .node-item code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-weight: 600; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; padding: 0.2em 0.4em; margin: 0; }
.node-summary .node-group .node-item pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; padding: 8px 12px 8px 12px; overflow: auto; line-height: 1.45; background-color: rgba(27, 31, 35, 0.05); border-radius: 8px; border: 1px solid rgba(27, 31, 35, 0.05); white-space: pre-wrap; word-wrap: break-word; padding: 4px 8px 4px 8px; }
.node-summary .node-group .node-item .group { margin-top: 8px; background-color: rgba(27, 31, 35, 0.05); border-radius: 8px; border: 1px solid rgba(27, 31, 35, 0.04); }
.node-summary .node-group .node-item .group-property { font-size: 10px; padding: 4px 8px 4px 8px; }
.node-summary .node-group .node-item .group-border { border-top: 1px solid rgba(27, 31, 35, 0.04); }
.node-summary .node-group .node-item .group code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(0, 0, 0, 0); padding: 0; margin: 0; border: 0; }
.node-summary .node-group .node-item .group pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(0, 0, 0, 0); margin: 0; padding: 4px 8px 4px 8px; border: 0; }

</style>
<div class='node-summary'>
<div class='node-group'>
{{#if operator}}
<h1>{{{operator}}}{{#if documentation}} <a id='documentation-button' class='documentation-button'>?</a>{{/if}}</h1>
{{/if}}
{{#if name}}
<div class='node-item'><b>name</b><br><pre>{{{name}}}</pre></div>
{{/if}}
{{#if description}}
<div class='node-item'><b>description</b><br><pre>{{{description}}}</pre></div>
{{/if}}
{{#if domain}}
<div class='node-item'><b>domain</b><br><pre>{{{domain}}}</pre></div>
{{/if}}
</div>

{{#if attributes}}
<h2>Attributes</h2>
{{/if}}
<div class='node-group'>
{{#attributes}}
<div class='node-item'>    
<b>{{{name}}}{{#if type}}: {{/if}}</b>{{#if type}}<code>{{{type}}}</code>{{/if}}<br>
{{#if description}}
{{{description}}}
{{/if}}
<pre>{{{value}}}</pre>
</div>
{{/attributes}}
</div>

{{#if inputs}}
<h2>Inputs</h2>
<div class='node-group'>
{{#inputs}}
<div class='node-item'>
<b>{{{name}}}{{#if type}}: {{/if}}</b>{{#if type}}<code>{{{type}}}</code>{{/if}}
{{#connections}}
<div class='group'>
<div class='group-property'>
connection: <b>{{{id}}}</b>
{{#if initializer}}
{{#if initializer.title}}
<div style='float: right;'>{{initializer.title}}</div>
{{/if}}
{{/if}}
</div>
{{#if type}}
<div class='group-border'></div>
<div class='group-property'>
type: <code><b>{{{type}}}</b></code>
</div>
{{/if}}
</div>
{{/connections}}
</div>
{{/inputs}}
</div>
{{/if}}

{{#if outputs}}
<h2>Outputs</h2>
<div class='node-group'>
{{#outputs}}
<div class='node-item'>
<b>{{{name}}}{{#if type}}: {{/if}}</b>{{#if type}}<code><b>{{{type}}}</b></code>{{/if}}
{{#connections}}
<div class='group'>
<div class='group-property'>
connection: <b>{{{id}}}</b>
</div>
{{#if type}}
<div class='group-border'></div>
<div class='group-property'>
type: <code><b>{{{type}}}</b></code>
</div>
{{/if}}
</div>
{{/connections}}
</div>
{{/outputs}}
</div>
{{/if}}

{{#if dependencies}}
<h2>Control Dependencies</h2>
<div class='node-group'>
<div class='node-item'>
{{#dependencies}}
<div class='group'>
<div class='group-property'>
connection: <b>{{{id}}}</b>
</div>
{{#if name}}
<div class='group-border'></div>
<div class='group-property'>
name: <b>{{{name}}}</b>
</div>
{{/if}}
{{#if operator}}
<div class='group-border'></div>
<div class='group-property'>
operator: <b>{{{operator}}}</b>
</div>
{{/if}}
</div>
{{/dependencies}}
</div>
</div>
{{/if}}

</div>
`;