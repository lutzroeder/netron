/*jshint esversion: 6 */

var itemsTemplate = `
<style type='text/css'>

.items {
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}
.item {
    margin-bottom: 20px;
}
.item b {
    font-weight: 600;
}
.item h1 {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.25;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 16px;
}
.item code {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 10px;
    background-color: rgba(27, 31, 35, 0.05);
    padding: 0.2em 0.4em;
    margin: 0;
    border-radius: 3px
}
.item pre {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 11px;
    padding: 8px 12px 8px 12px;
    overflow: auto;
    line-height: 1.45;
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    white-space: pre-wrap;
    word-wrap: break-word;  
}
</style>
<div class='items'>
{{#items}}
<div class='item'>    
<b>{{{name}}}{{#if type}}: {{/if}}</b>{{#if type}}<code>{{{type}}}</code>{{/if}}<br>
{{#if quantization}}
<pre>{{{quantization}}}</pre>
{{/if}}
{{#if doc}}
{{{doc}}}
{{/if}}
<pre>{{{value}}}</pre>
</div>
{{/items}}
</div>
`;

var operatorTemplate = `
<style type='text/css'>

.documentation {
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}
.documentation h1 {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.25;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 16px;
}
.documentation h2 {
    font-weight: 600;
    font-size: 12px;
    line-height: 1.25;
    margin-bottom: 16px;
    border-bottom: 1px solid #eaecef
}
.documentation h3 {
    font-weight: 600;
    font-size: 12px;
    line-height: 1.25;
}
.documentation code {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 10px;
    background-color: rgba(27, 31, 35, 0.05);
    padding: 0.2em 0.4em;
    margin: 0;
    border-radius: 3px
}
.documentation pre {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 11px;
    padding: 16px;
    overflow: auto;
    line-height: 1.45;
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px
}
.documentation tt {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-weight: 600;
    font-size: 85%;
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    padding: 0.2em 0.4em;
    margin: 0;
}
.documentation dl dt {
    font-size: 12px;
    font-weight: 600;
    padding: 0;
    margin-top: 16px;
}
.documentation dd {
    padding: 0 16px;
    margin-left: 0;
    margin-bottom: 16px;
}
</style>

<div class='documentation'>

<h1>{{{name}}}</h1>
{{#if doc}}
{{{doc}}}
{{/if}}

{{#if attributes}}
<h2>Attributes</h2>
<dl>
{{#attributes}}
<dt><tt>{{{name}}}</tt> : {{{type}}}</dt>
<dd>{{{description}}}</dd>
{{/attributes}}
</dl>
{{/if}}

{{#if inputs}}
<h2>Inputs{{#if inputs_range}} ({{{inputs_range}}}){{/if}}</h2>
<dl>
{{/if}}
{{#inputs}}
<dt><tt>{{{name}}}</tt> {{#if option}}({{{option}}}){{/if}}: {{{typeStr}}}</dt>
<dd>{{{description}}}</dd>
{{/inputs}}
</dl>

{{#if outputs.length}}
<h2>Outputs{{#if outputs_range}} ({{{outputs_range}}}){{/if}}</h2>
<dl>
{{/if}}
{{#outputs}}
<dt><tt>{{{name}}}</tt> {{#if option}}({{{option}}}){{/if}}: {{{typeStr}}}</dt>
<dd>{{{description}}}</dd>
{{/outputs}}
</dl>

{{#if type_constraints}}
<h2>Type Constraints</h2>
<dl>
{{#type_constraints}}
<dt><tt>{{{type_param_str}}}</tt>: {{{allowed_type_strs_display}}}</dt>
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
.summary {
    font-family: 'Open Sans', --apple-system, "Helvetica Neue", Helvetica, Arial, sans-serf;
    font-size: 12px;
    line-height: 1.5;
    overflow: hidden;
    width: 500;
    margin: auto;
}

.summary h1 {
    font-family: 'Open Sans', --apple-system, "Helvetica Neue", Helvetica, Arial, sans-serf;
    font-weight: 600;
    font-size: 12px;
    margin: 0;
    color: #666;
    letter-spacing: 0.5px;
    padding: 10px 0px 0px 0px;
    margin: 20px 0px 0px 0px;
    -webkit-user-select: none;
    -moz-user-select: none;
    user-select: none;
}

.summary .section {
    margin-top: 10px;
    margin-bottom: 10px;
    padding: 10px;
    overflow-y: auto;
    position: relative;
    border: 1px solid none;
}

.summary .section table code {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 10px;
    background-color: rgba(27, 31, 35, 0.05);
    padding: 0.2em 0.4em;
    margin: 0;
    border-radius: 3px;
    background-color: #d6d6d6;
}

.summary .border {
    border-radius: 10px;
    border: 1px solid #ccc;
}

.summary .section table { 
    color: #777;
    float: left;
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    border-spacing: 0;
    line-height: 150%;
}

.summary .section table td:first-child {
    font-weight: 600;
    width: 75px;
    vertical-align: top;
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
}

.summary .section table td { user-select: text; -webkit-user-select: text; -moz-user-select: text; }
</style>
<div class='summary'>
<h1>MODEL PROPERTIES</h1>
<div class='border section'>
<table>
{{#properties}}
<tr><td>{{{name}}}</td><td>{{{value}}}</td></tr>
{{/properties}}
</table>
</div>
<h1>GRAPHS</h1>
{{#graphs}}
<div class='border section'>
<table>
{{#if name}}
<tr><td>Name</td><td>{{{name}}}</td></tr>
{{/if}}
{{#if version}}
<tr><td>Version</td><td>{{{version}}}</td></tr>
{{/if}}
{{#if tags}}
<tr><td>Tags</td><td>{{{tags}}}</td></tr>
{{/if}}
<tr>
{{#if inputs}}
<td>Inputs</td>
<td>
{{#inputs}}
<code>{{{type}}}</code> : {{{name}}}<br>
{{/inputs}}
</td>
</tr>
{{/if}}
{{#if outputs}}
<tr>
<td>Outputs</td>
<td>
{{#outputs}}
<code>{{{type}}}</code> : {{{name}}}<br>
{{/outputs}}
</td>
</tr>
{{/if}}
{{#if description}}
<tr><td>{{{Description}}}</td><td>{{{description}}}</td></tr>
{{/if}}
</table>
<button style='float: right; width: 80px;' onclick='javascript:updateActiveGraph("{{{name}}}");'>View</button>
</div>
{{/graphs}}
</div>
`;