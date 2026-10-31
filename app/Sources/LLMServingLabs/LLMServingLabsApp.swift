import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct LLMServingLabsApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = CourseStore()
    @StateObject private var progress = ProgressStore()

    var body: some Scene {
        WindowGroup("LLM Serving Labs") {
            ContentView()
                .environmentObject(store)
                .environmentObject(progress)
                .frame(minWidth: 960, minHeight: 640)
        }
        .defaultSize(width: 1240, height: 820)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Reload Course") { store.load() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                Divider()
                Button("Reset Progress…") { progress.confirmReset() }
            }
        }
    }
}
