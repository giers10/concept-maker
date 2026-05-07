import MarkdownIt from "markdown-it";

const EXTRA_BLANK_LINE_MARKER = "\uE000CONCEPT_MAKER_EXTRA_BLANK_LINE\uE000";

const escapeHtml = (value = "") =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const appendClass = (token: any, className: string) => {
  const existing = token.attrGet("class");
  token.attrSet("class", existing ? `${existing} ${className}` : className);
};

const removeAttr = (token: any, attrName: string) => {
  const index = token.attrIndex(attrName);
  if (index >= 0) token.attrs.splice(index, 1);
};

const alignmentClassFromToken = (token: any) => {
  const style = token.attrGet("style") || "";
  const match = style.match(/text-align:\s*(left|right|center)/i);
  return `md-align-${match?.[1]?.toLowerCase() || "left"}`;
};

const safeLink = (hrefRaw: string) => {
  const href = (hrefRaw || "").trim();
  if (!href) return "";
  if (href.startsWith("//")) return "";
  if (/^(https?|mailto|tel):/i.test(href)) return href;
  if (/^[./#]/.test(href)) return href;
  if (!/^[a-z][a-z0-9+.-]*:/i.test(href)) return href;
  return "";
};

const preserveExtraBlankLines = (source: string) =>
  source.replace(/([^\n])(\n{3,})([^\n])/g, (_match, before, newlines, after) => {
    const markers = Array.from(
      { length: newlines.length - 2 },
      () => EXTRA_BLANK_LINE_MARKER
    ).join("\n\n");
    return `${before}\n\n${markers}\n\n${after}`;
  });

const restoreExtraBlankLines = (html: string) => {
  const markerParagraph = new RegExp(
    `<p class="md-paragraph">${escapeRegExp(EXTRA_BLANK_LINE_MARKER)}</p>\\n?`,
    "g"
  );
  return html.replace(markerParagraph, "<br />");
};

const cleanMarkdownForPreview = (source: string) =>
  source
    .replace(/<think(?:ing)?>[\s\S]*?(?:<\/think(?:ing)?>|$)/gi, "")
    .replace(/[\u00a0\u202f\u2007]/g, " ");

function balanceStreamingCodeFence(markdownSource: string) {
  const lines = markdownSource.split(/\r?\n/);
  let open: { fenceChar: string; fenceLen: number } | null = null;

  for (const line of lines) {
    if (!open) {
      const match = line.match(/^\s*([`~]{3,})([^\s]*)?.*$/);
      if (match) {
        open = { fenceChar: match[1][0], fenceLen: match[1].length };
      }
      continue;
    }

    const closeFence = new RegExp(`^\\s*(${open.fenceChar}{${open.fenceLen},})\\s*$`);
    if (closeFence.test(line)) {
      open = null;
    }
  }

  if (!open) return markdownSource;

  const virtualFence = open.fenceChar.repeat(open.fenceLen);
  return markdownSource.endsWith("\n")
    ? markdownSource + virtualFence
    : `${markdownSource}\n${virtualFence}`;
}

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: false
});

markdown.linkify.set({ fuzzyEmail: false, fuzzyIP: false });
markdown.validateLink = (href: string) => Boolean(safeLink(href));

markdown.renderer.rules.paragraph_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-paragraph");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.heading_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  const level = tokens[idx].tag.replace(/^h/i, "") || "1";
  appendClass(tokens[idx], `md-heading md-heading--${level}`);
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.hr = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-hr");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.bullet_list_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-list md-list--unordered");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.ordered_list_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-list md-list--ordered");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.list_item_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-list__item");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.blockquote_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-blockquote");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.table_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], "md-table");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.thead_open = (
  tokens: any[],
  idx: number,
  options: any,
  env: any,
  self: any
) => {
  env.inTableHead = true;
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.thead_close = (
  tokens: any[],
  idx: number,
  options: any,
  env: any,
  self: any
) => {
  env.inTableHead = false;
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.tr_open = (
  tokens: any[],
  idx: number,
  options: any,
  env: any,
  self: any
) => {
  appendClass(
    tokens[idx],
    env.inTableHead ? "md-table__row md-table__row--head" : "md-table__row"
  );
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.th_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], `md-table__head-cell ${alignmentClassFromToken(tokens[idx])}`);
  removeAttr(tokens[idx], "style");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.td_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  appendClass(tokens[idx], `md-table__cell ${alignmentClassFromToken(tokens[idx])}`);
  removeAttr(tokens[idx], "style");
  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.strong_open = () => "<b>";
markdown.renderer.rules.strong_close = () => "</b>";
markdown.renderer.rules.em_open = () => "<i>";
markdown.renderer.rules.em_close = () => "</i>";
markdown.renderer.rules.softbreak = () => "<br />";
markdown.renderer.rules.hardbreak = () => "<br />";

markdown.renderer.rules.code_inline = (tokens: any[], idx: number) =>
  `<code class="md-inline-code">${escapeHtml(tokens[idx].content)}</code>`;

markdown.renderer.rules.fence = (tokens: any[], idx: number) => {
  const info = tokens[idx].info.trim();
  const languageClass = info.toLowerCase().replace(/[^a-z0-9_-]/g, "") || "code";
  return `<pre class="md-codeblock"><code class="language-${languageClass}">${escapeHtml(tokens[idx].content).replace(
    /\n$/,
    ""
  )}</code></pre>`;
};

markdown.renderer.rules.code_block = (tokens: any[], idx: number) =>
  `<pre class="md-codeblock"><code>${escapeHtml(tokens[idx].content).replace(
    /\n$/,
    ""
  )}</code></pre>`;

markdown.renderer.rules.link_open = (
  tokens: any[],
  idx: number,
  options: any,
  _env: any,
  self: any
) => {
  const href = safeLink(tokens[idx].attrGet("href") || "");

  if (!href) return "";

  tokens[idx].attrSet("href", href);
  tokens[idx].attrSet("class", "md-link");

  if (/^https?:\/\//i.test(href)) {
    tokens[idx].attrSet("target", "_blank");
    tokens[idx].attrSet("rel", "noreferrer noopener");
  }

  return self.renderToken(tokens, idx, options);
};

markdown.renderer.rules.image = (
  tokens: any[],
  idx: number,
  options: any,
  env: any,
  self: any
) => {
  const token = tokens[idx];
  const src = safeLink(token.attrGet("src") || "");
  if (!src) return escapeHtml(token.content || "");

  token.attrSet("src", src);
  token.attrSet("alt", self.renderInlineAsText(token.children, options, env));
  token.attrSet("loading", "lazy");
  appendClass(token, "md-image");

  return self.renderToken(tokens, idx, options);
};

export function markdownToHTML(text: string) {
  let source = cleanMarkdownForPreview(String(text ?? ""));
  source = balanceStreamingCodeFence(source);
  source = preserveExtraBlankLines(source);

  const html = markdown.render(source, {});
  return restoreExtraBlankLines(html).replace(/\n+$/g, "");
}
