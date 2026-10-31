import AppKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: CourseStore
    @EnvironmentObject private var progress: ProgressStore
    @AppStorage("lesson") private var lessonID = ""
    @AppStorage("mode") private var modeRaw = Mode.read.rawValue
    @AppStorage("study") private var studyRaw = Study.learn.rawValue
    @AppStorage("scope") private var scopeRaw = Scope.lesson.rawValue
    @AppStorage("practice") private var practiceRaw = Practice.all.rawValue
    @AppStorage("shuffle") private var shuffle = false
    @StateObject private var searchState = SearchState()
    @StateObject private var session = CardSession()

    private var search: String { searchState.text }

    private var mode: Mode { Mode(rawValue: modeRaw) ?? .read }
    private var lesson: Lesson? { store.lesson(lessonID.isEmpty ? nil : lessonID) }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 260, ideal: 310, max: 420)
        } detail: {
            detail
        }
        .searchable(text: $searchState.text, placement: .sidebar, prompt: "Search lessons")
        .toolbar { toolbar }
        .navigationTitle(lesson?.title ?? store.course.title)
        .onAppear(perform: rebuild)
        .onChange(of: lessonID) { rebuild() }
        .onChange(of: modeRaw) { rebuild() }
        .onChange(of: studyRaw) { rebuild() }
        .onChange(of: scopeRaw) { rebuild() }
        .onChange(of: practiceRaw) { rebuild() }
        .onChange(of: shuffle) { rebuild() }
        .onChange(of: store.course.version) { rebuild() }
    }

    private func rebuild() {
        session.rebuild(
            store.cards(scope: Scope(rawValue: scopeRaw) ?? .lesson, lessonID: lesson?.id),
            practice: Practice(rawValue: practiceRaw) ?? .all, shuffle: shuffle, progress: progress)
    }

    private func open(_ id: String?, read: Bool = true) {
        lessonID = id ?? ""
        if read { modeRaw = Mode.read.rawValue }
    }

    // MARK: Sidebar

    private var selection: Binding<String?> {
        Binding(get: { lessonID.isEmpty ? "home" : lessonID },
                set: { lessonID = ($0 == nil || $0 == "home") ? "" : $0! })
    }

    private var sidebar: some View {
        List(selection: selection) {
            if search.trimmingCharacters(in: .whitespaces).isEmpty {
                Label("Contents and progress", systemImage: "house").tag("home")
                ForEach(store.course.tracks) { track in
                    Section {
                        ForEach(track.lessons) { lessonRow($0) }
                    } header: {
                        HStack {
                            Text(track.title)
                            Spacer()
                            Text("\(track.lessons.filter { progress.done.contains($0.id) }.count)/\(track.lessons.count)")
                                .monospacedDigit()
                        }
                    }
                }
            } else {
                let hits = store.lessons.filter { store.matches($0, search) }
                Section("Results (\(hits.count))") {
                    ForEach(hits) { lessonRow($0) }
                }
            }
        }
        .listStyle(.sidebar)
    }

    private func lessonRow(_ lesson: Lesson) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: progress.done.contains(lesson.id) ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(progress.done.contains(lesson.id) ? Color.green : Color.secondary)
                .font(.caption)
            Text(lesson.title).lineLimit(2)
            Spacer(minLength: 4)
            if !lesson.cards.isEmpty {
                Text("\(lesson.cards.count)").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .tag(lesson.id)
    }

    // MARK: Toolbar

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            Picker("Mode", selection: $modeRaw) {
                Text("Read").tag(Mode.read.rawValue)
                Text("Cards").tag(Mode.cards.rawValue)
            }
            .pickerStyle(.segmented)
            .frame(width: 160)
        }
        if mode == .cards {
            ToolbarItemGroup(placement: .primaryAction) {
                Picker("Study", selection: $studyRaw) {
                    Text("Learn").tag(Study.learn.rawValue)
                    Text("Test").tag(Study.test.rawValue)
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
                Picker("Cards from", selection: $scopeRaw) {
                    Text("This lesson").tag(Scope.lesson.rawValue)
                    Text("This track").tag(Scope.track.rawValue)
                    Text("Whole course").tag(Scope.all.rawValue)
                }
                .frame(width: 140)
                .disabled(lesson == nil)
                Menu {
                    Picker("Practice", selection: $practiceRaw) {
                        Text("All cards").tag(Practice.all.rawValue)
                        Text("Not yet known").tag(Practice.notKnown.rawValue)
                        Text("Review only").tag(Practice.review.rawValue)
                    }
                    .pickerStyle(.inline)
                    Toggle("Shuffle", isOn: $shuffle)
                    Button("Restart") { rebuild() }
                } label: {
                    Label("Options", systemImage: "slider.horizontal.3")
                }
            }
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        if store.course.tracks.isEmpty {
            ContentUnavailableView("Course not loaded", systemImage: "exclamationmark.triangle",
                                   description: Text(store.errorMessage ?? ""))
        } else if mode == .cards {
            CardsView(session: session, rebuild: rebuild, openLesson: { open($0) },
                      setScope: { scopeRaw = $0.rawValue }, setStudy: { studyRaw = $0.rawValue },
                      showAll: { practiceRaw = Practice.all.rawValue },
                      study: Study(rawValue: studyRaw) ?? .learn, hasLesson: lesson != nil,
                      scope: Scope(rawValue: scopeRaw) ?? .lesson, practice: Practice(rawValue: practiceRaw) ?? .all)
        } else if let lesson {
            reader(lesson)
        } else {
            HomeView(open: { open($0) })
        }
    }

    private func reader(_ lesson: Lesson) -> some View {
        let track = store.trackOf[lesson.id]
        let done = progress.done.contains(lesson.id)
        return VStack(spacing: 0) {
            HTMLView(html: "<div class=\"eyebrow\">\(track?.title ?? "")</div><h1>\(lesson.title)</h1>" + lesson.html,
                     openLesson: { open($0) })
            Divider()
            HStack(spacing: 10) {
                Button { open(store.neighbor(of: lesson.id, -1)?.id) } label: { Label("Previous", systemImage: "chevron.left") }
                    .keyboardShortcut(.leftArrow, modifiers: .command)
                    .disabled(store.neighbor(of: lesson.id, -1) == nil)
                Spacer()
                Button(done ? "✓ Read" : "Mark as read") { progress.toggleDone(lesson.id) }
                    .tint(done ? .green : nil)
                    .keyboardShortcut("d", modifiers: .command)
                if !lesson.cards.isEmpty {
                    Button("Study \(lesson.cards.count) cards") {
                        scopeRaw = Scope.lesson.rawValue
                        modeRaw = Mode.cards.rawValue
                    }
                    .buttonStyle(.borderedProminent)
                }
                Button("View on GitHub") { if let url = URL(string: lesson.url) { NSWorkspace.shared.open(url) } }
                Spacer()
                Button { open(store.neighbor(of: lesson.id, 1)?.id) } label: { Label("Next", systemImage: "chevron.right") }
                    .labelStyle(TrailingIcon())
                    .keyboardShortcut(.rightArrow, modifiers: .command)
                    .disabled(store.neighbor(of: lesson.id, 1) == nil)
            }
            .controlSize(.large)
            .padding(12)
        }
    }
}

/// The Command Line Tools toolchain lacks the SwiftUI macro plugin behind `@State`.
final class SearchState: ObservableObject {
    @Published var text = ""
}

private struct TrailingIcon: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) { configuration.title; configuration.icon }
    }
}

struct HomeView: View {
    @EnvironmentObject private var store: CourseStore
    @EnvironmentObject private var progress: ProgressStore
    let open: (String?) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text(store.course.title).font(.largeTitle.bold())
                Text("From one GPU to a global service: systems foundations, vLLM, SGLang and llm-d labs on an 8xB200 node, and cloud deployment. Read a lesson, then test yourself with its cards.")
                    .foregroundStyle(.secondary)
                let known = store.allCards.filter { progress.known.contains($0.id) }.count
                HStack(spacing: 14) {
                    Button(progress.done.isEmpty ? "Start the course" : "Continue") {
                        open((store.lessons.first { !progress.done.contains($0.id) } ?? store.lessons.first)?.id)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    Text("\(progress.done.count) of \(store.lessons.count) lessons read · \(known) of \(store.allCards.count) cards known")
                        .foregroundStyle(.secondary)
                }
                ForEach(store.course.tracks) { track in
                    trackSection(track)
                }
            }
            .padding(32)
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
    }

    private func trackSection(_ track: Track) -> some View {
        let done = track.lessons.filter { progress.done.contains($0.id) }.count
        let cards = track.lessons.flatMap(\.cards)
        let known = cards.filter { progress.known.contains($0.id) }.count
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(track.title).font(.title2.bold())
                Spacer()
                Text("\(done)/\(track.lessons.count) read" + (cards.isEmpty ? "" : " · \(known)/\(cards.count) known"))
                    .foregroundStyle(.secondary).monospacedDigit()
            }
            Text(track.blurb).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 0) {
                ForEach(track.lessons) { lesson in
                    Button { open(lesson.id) } label: {
                        HStack {
                            Image(systemName: progress.done.contains(lesson.id) ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(progress.done.contains(lesson.id) ? Color.green : Color.secondary)
                            Text(lesson.title)
                            Spacer()
                        }
                        .padding(.vertical, 7)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    Divider()
                }
            }
        }
        .padding(.top, 14)
    }
}
