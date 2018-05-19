/*jshint esversion: 6 */

var operatorTemplate = `
<style type='text/css'>
.documentation { font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; font-size: 12px; line-height: 1.5; margin: 0; }
.documentation h1 { font-weight: 600; font-size: 14px; line-height: 1.25; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 0; margin-bottom: 16px; }
.documentation h2 { font-weight: 600; font-size: 12px; line-height: 1.25; margin-bottom: 16px; border-bottom: 1px solid #eaecef; }
.documentation h3 { font-weight: 600; font-size: 12px; line-height: 1.25; }
.documentation p { margin-top: 2px; margin-bottom: 2px; }
.documentation a { color: #237; }
.documentation code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); padding: 0.2em 0.4em; margin: 0; border-radius: 3px }
.documentation pre { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 11px; padding: 16px; overflow: auto; line-height: 1.45; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; }
.documentation pre code { font-size: 11px; padding: 16px; line-height: 1.45; background-color: transparent; padding: 0; border-radius: 0; }
.documentation tt { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-weight: 600; font-size: 85%; background-color: rgba(27, 31, 35, 0.05); border-radius: 3px; padding: 0.2em 0.4em; margin: 0; }
.documentation dl dt { font-size: 12px; font-weight: 600; padding: 0; margin-top: 16px; }
.documentation dd { padding: 0 16px; margin-left: 0; margin-bottom: 16px; }
.documentation ul { margin-top: 6px; margin-bottom: 6px; padding-left: 20px; }
.documentation li { }
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

{{#if examples}}
<h2>Examples</h2>
{{#examples}}
<h3>{{{summary}}}</h3>
<pre>{{{code}}}</pre>
{{/examples}}
{{/if}}

{{#if references}}
<h2>References</h2>
<ul>
{{#references}}
<li>{{{description}}}</li>
{{/references}}
</ul>
{{/if}}

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
.summary h1 { font-family: inherit; font-weight: 600; font-size: 12px; margin: 0; color: #333; letter-spacing: 0.5px; padding: 10px 0px 0px 0px; margin: 10px 0px 0px 0px; -webkit-user-select: none; -moz-user-select: none; user-select: none; }
.summary .section { color: #777; margin-top: 10px; margin-bottom: 10px; padding: 10px; overflow-y: auto; position: relative; border: 1px solid none; border-radius: 10px; border: 1px solid #ccc; }
.summary .section .property { margin: 2px 0 2px 0; overflow: hidden; width: 100% !important; }
.summary .section .property .name { float: left; clear: left; width: 80px; min-width: 80px; font-weight: 600; }
.summary .section .property .value { overflow: hidden; }
.summary .section .property .value b { font-weight: 600; }
.summary .section .property .value input { font-family: inherit; font-size: inherit; color: inherit; background-color: inherit; width: 100%; margin: 0; padding: 0; border: 0; outline: none; text-overflow: ellipsis; }
.summary .section .property .value code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 10px; background-color: rgba(27, 31, 35, 0.05); padding: 0.2em 0.4em; margin: 0; border-radius: 3px; background-color: #d6d6d6; }
</style>

<div class='summary'>

<div class='section'>
{{#properties}}
<div class='property'>
<div class='name'>{{name}}</div>
<div class='value'>{{value}}</div>
</div>
{{/properties}}
</div>

<h1>GRAPHS</h1>
{{#graphs}}
<div class='section'>
{{#if name}}
<div class='property'>
<div class='name'>Name</div>
<div class='value'><input type='text' value='{{name}}'/></div>
</div>
{{/if}}
{{#if version}}
<div class='property'>
<div class='name'>Version</div>
<div class='value'>{{version}}</div>
</div>
{{/if}}
{{#if type}}
<div class='property'>
<div class='name'>Type</div>
<div class='value'>{{type}}</div>
</div>
{{/if}}
{{#if tags}}
<div class='property'>
<div class='name'>Tags</div>
<div class='value'>{{tags}}</div>
</div>
{{/if}}
{{#if description}}
<div class='property'>
<div class='name'>Description</div>
<div class='value'>{{description}}</div>
</div>
{{/if}}
<div class='property'>
{{#if inputs}}
<div class='name'>Inputs</div>
<div class='value'>
{{#inputs}}
<b>{{name}}</b>{{#if type}}: <code>{{type}}</code>{{/if}}<br>
{{#if description}}<div style='margin-left: 20px'>{{description}}</div>{{/if}}
{{/inputs}}
</div>
</div>
{{/if}}
{{#if outputs}}
<div class='property'>
<div class='name'>Outputs</div>
<div class='value'>
{{#outputs}}
<b>{{name}}</b>{{#if type}}: <code>{{type}}</code>{{/if}}<br>
{{#if description}}<div style='margin-left: 20px'>{{description}}</div>{{/if}}
{{/outputs}}
</div>
</div>
{{/if}}
<div class='property'>
<div class='value'>
<button style='float: right; width: 80px; margin: 2px 0 0 0;' onclick='javascript:updateActiveGraph("{{{name}}}");'>View</button>
</div>
</div>
{{/graphs}}
</div>
`;
