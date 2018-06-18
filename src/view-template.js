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
