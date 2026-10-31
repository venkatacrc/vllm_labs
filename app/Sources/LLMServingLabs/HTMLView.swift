import AppKit
import SwiftUI
import WebKit

/// Shows pre-rendered lesson HTML. Links to other lessons (`#/lesson/<id>`) call `openLesson`;
/// web links open in the default browser.
struct HTMLView: NSViewRepresentable {
    let html: String
    var openLesson: (String) -> Void = { _ in }

    static let base = URL(string: "https://venkatacrc.github.io/vllm_labs/")!

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> WKWebView {
        let view = WKWebView()
        view.navigationDelegate = context.coordinator
        view.setValue(false, forKey: "drawsBackground")
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        context.coordinator.parent = self
        guard context.coordinator.loaded != html else { return }
        context.coordinator.loaded = html
        view.loadHTMLString(Self.page(html), baseURL: Self.base)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var parent: HTMLView
        var loaded: String?

        init(_ parent: HTMLView) { self.parent = parent }

        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard action.navigationType == .linkActivated, let url = action.request.url else {
                decisionHandler(.allow)
                return
            }
            decisionHandler(.cancel)
            if let fragment = url.fragment, fragment.hasPrefix("/lesson/") {
                parent.openLesson(String(fragment.dropFirst("/lesson/".count)))
            } else {
                NSWorkspace.shared.open(url)
            }
        }
    }

    static func page(_ body: String) -> String {
        """
        <!DOCTYPE html><html><head><meta charset="utf-8">
        <style>\(css)</style></head><body>\(body)</body></html>
        """
    }

    static let css = """
    :root { color-scheme: light dark; --accent: #4c5fd5; --text: #1a1c24; --muted: #686d7c; --line: #e1e3ec; --code-bg: #f0f1f6;
            --green: #2e9e5b; }
    @media (prefers-color-scheme: dark) { :root { --accent: #8191ff; --text: #eceef6; --muted: #a0a5b6; --line: #2f3240; --code-bg: #252834; } }
    html, body { margin: 0; background: transparent; color: var(--text); }
    body { font: 15px/1.6 -apple-system, system-ui, sans-serif; padding: 26px 34px 40px; max-width: 900px; margin: 0 auto; overflow-wrap: break-word; }
    a { color: var(--accent); }
    h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 14px; }
    h2 { font-size: 1.25rem; margin: 1.6em 0 0.5em; }
    h3 { font-size: 1.05rem; margin: 1.3em 0 0.4em; }
    p, ul, ol { margin: 0 0 0.9em; }
    ul, ol { padding-left: 1.4em; }
    li { margin: 0.2em 0; }
    blockquote { margin: 0 0 1em; padding: 4px 14px; border-left: 3px solid var(--accent); color: var(--muted); }
    code { padding: 1px 5px; border-radius: 5px; background: var(--code-bg); font: 0.86em ui-monospace, "SF Mono", Menlo, monospace; }
    pre { margin: 0 0 1em; padding: 12px 14px; border-radius: 10px; background: var(--code-bg); overflow-x: auto; line-height: 1.4; }
    pre code { padding: 0; background: none; font-size: 12.5px; white-space: pre; }
    .table-wrap { overflow-x: auto; margin: 0 0 1em; }
    table { border-collapse: collapse; font-size: 0.9rem; min-width: 100%; }
    th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { border-bottom-width: 2px; }
    .eyebrow { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent); }
    .question { font-size: 1.45rem; font-weight: 600; line-height: 1.4; margin: 6px 0 0; }
    .answer { margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 1.05rem; }
    .hint { margin-top: 40px; color: var(--muted); text-align: center; }
    """
}
