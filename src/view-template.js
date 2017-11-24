
var attributesTemplate = `
<style type='text/css'>

.attributes {
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}
.attribute {
    margin-bottom: 20px;
}
.attribute h1 {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.25;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 16px;
}
.attribute code {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 10px;
    background-color: rgba(27, 31, 35, 0.05);
    padding: 0.2em 0.4em;
    margin: 0;
    border-radius: 3px
}
.attribute pre {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 11px;
    padding: 12px;
    overflow: auto;
    line-height: 1.45;
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    white-space: pre-wrap;
    word-wrap: break-word;  
}
</style>

<div class='attributes'>

{{#attributes}}
<div class='attribute'>    
<h1>{{{name}}}</h1>
{{#if doc}}
{{{doc}}}
{{/if}}
<pre>{{{value}}}</pre>
</div>
{{/attributes}}

</div>
`;

var initializerTemplate = `
<style type='text/css'>

.initializer {
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}
.initializer h1 {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.25;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 16px;
}
.initializer h2 {
    font-weight: 600;
    font-size: 12px;
    line-height: 1.25;
    margin-bottom: 16px;
    border-bottom: 1px solid #eaecef
}
.initializer pre {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 11px;
    padding: 16px;
    overflow: auto;
    line-height: 1.45;
    background-color: rgba(27, 31, 35, 0.05);
    border-radius: 3px;
    white-space: pre-wrap;
    word-wrap: break-word;  
}
</style>

<div class='initializer'>
<h1>{{{name}}}</h1>
{{#if doc}}
{{{doc}}}
{{/if}}
<h2>type</h2>
<pre>{{{type}}}</pre>
<h2>value</h2>
<pre>{{{value}}}</pre>
</div>
`;

var propertiesTemplate = `
<style type='text/css'>

.properties {
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}

.properties h1 {
    font-weight: 600;
    font-size: 14px;
    line-height: 1.25;
    padding-top: 5px;
    padding-left: 0px;
    padding-right: 0px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 16px;
}

.properties table { 
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 12px;
    margin-bottom: 30px;
    border-spacing: 0;
}

.properties table td:first-child {
    font-weight: 600;
    width: 100px;
}

</style>

<div class='properties'>

{{#groups}}
<h1>{{{name}}}</h1>
<table id='{{{name}}}' class='content'>
{{#properties}}
<tr>
<td>{{{name}}}</td>
<td>{{{value}}}</td>
</tr>
{{/properties}}
</table>
{{/groups}}

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

<h1>{{{op_type}}}</h1>
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
<h2>Inputs</h2>
<dl>
{{/if}}
{{#inputs}}
<dt><tt>{{{name}}}</tt> {{#if optional}}(optional){{/if}}: {{{typeStr}}}</dt>
<dd>{{{description}}}</dd>
{{/inputs}}
</dl>

{{#if outputs.length}}
<h2>Outputs</h2>
<dl>
{{/if}}
{{#outputs}}
<dt><tt>{{{name}}}</tt> : {{{typeStr}}}</dt>
<dd>{{{description}}}</dd>
{{/outputs}}
</dl>

{{#if type_constraints}}
<h2>Type Constraints</h2>
{{/if}}
{{#type_constraints}}
<dl>
<dt><tt>{{{type_param_str}}}</tt>: {{{allowed_type_strs_display}}}</dt>
<dd>{{{description}}}</dd>
</dl>
{{/type_constraints}}

{{#if snippets}}
<h2>Examples</h2>
{{/if}}
{{#snippets}}
<h3>{{{summary}}}</h3>
<pre>{{{code}}}</pre>
{{/snippets}}

</div>
`;