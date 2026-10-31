import AppKit
import Foundation

@MainActor
final class CourseStore: ObservableObject {
    @Published private(set) var course: Course = .empty
    @Published private(set) var origin = ""
    @Published var errorMessage: String?

    private(set) var lessons: [Lesson] = []
    private(set) var trackOf: [String: Track] = [:]
    private var searchText: [String: String] = [:]

    init() { load() }

    /// Prefers the copy inside the .app, then LLM_LABS_COURSE, then web/course.json found by
    /// walking up from the working directory or the executable (used by `swift run`).
    func load() {
        var candidates: [URL] = []
        if let url = Bundle.main.url(forResource: "course", withExtension: "json") { candidates.append(url) }
        if let env = ProcessInfo.processInfo.environment["LLM_LABS_COURSE"] { candidates.append(URL(fileURLWithPath: env)) }
        for start in [URL(fileURLWithPath: FileManager.default.currentDirectoryPath), Bundle.main.executableURL] {
            var dir = start?.standardizedFileURL
            for _ in 0..<8 {
                guard let current = dir else { break }
                candidates.append(current.appendingPathComponent("web/course.json"))
                dir = current.deletingLastPathComponent()
            }
        }
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            do {
                course = try JSONDecoder().decode(Course.self, from: Data(contentsOf: url))
                origin = url.path
                index()
                errorMessage = nil
                return
            } catch {
                errorMessage = "Could not read \(url.path): \(error.localizedDescription)"
            }
        }
        if errorMessage == nil {
            errorMessage = "course.json not found. Run python3 tools/build_course.py, then scripts/build_app.sh."
        }
    }

    private func index() {
        lessons = course.tracks.flatMap(\.lessons)
        trackOf = [:]
        searchText = [:]
        for track in course.tracks {
            for lesson in track.lessons { trackOf[lesson.id] = track }
        }
    }

    var allCards: [StudyCard] {
        lessons.flatMap { lesson in lesson.cards.map { StudyCard(card: $0, lesson: lesson) } }
    }

    func lesson(_ id: String?) -> Lesson? {
        guard let id else { return nil }
        return lessons.first { $0.id == id }
    }

    func neighbor(of id: String?, _ delta: Int) -> Lesson? {
        guard let id, let i = lessons.firstIndex(where: { $0.id == id }) else { return nil }
        let j = i + delta
        return lessons.indices.contains(j) ? lessons[j] : nil
    }

    func cards(scope: Scope, lessonID: String?) -> [StudyCard] {
        guard let lesson = lesson(lessonID) else { return allCards }
        switch scope {
        case .lesson: return lesson.cards.map { StudyCard(card: $0, lesson: lesson) }
        case .track:
            let track = trackOf[lesson.id]
            return allCards.filter { trackOf[$0.lesson.id]?.id == track?.id }
        case .all: return allCards
        }
    }

    func matches(_ lesson: Lesson, _ query: String) -> Bool {
        let terms = query.lowercased().split(separator: " ")
        if terms.isEmpty { return true }
        let text = searchText[lesson.id] ?? {
            let plain = (lesson.title + " " + lesson.html)
                .replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
                .lowercased()
            searchText[lesson.id] = plain
            return plain
        }()
        return terms.allSatisfy { text.contains($0) }
    }
}

@MainActor
final class ProgressStore: ObservableObject {
    @Published private(set) var known: Set<String>
    @Published private(set) var review: Set<String>
    @Published private(set) var done: Set<String>

    private let defaults = UserDefaults.standard

    init() {
        known = Set(defaults.stringArray(forKey: "knownCards") ?? [])
        review = Set(defaults.stringArray(forKey: "reviewCards") ?? [])
        done = Set(defaults.stringArray(forKey: "readLessons") ?? [])
    }

    func toggleKnown(_ id: String) {
        if known.remove(id) == nil {
            known.insert(id)
            review.remove(id)
        }
        save()
    }

    func toggleReview(_ id: String) {
        if review.remove(id) == nil {
            review.insert(id)
            known.remove(id)
        }
        save()
    }

    func markKnown(_ id: String) {
        known.insert(id)
        review.remove(id)
        save()
    }

    func markReview(_ id: String) {
        review.insert(id)
        known.remove(id)
        save()
    }

    func toggleDone(_ id: String) {
        if done.remove(id) == nil { done.insert(id) }
        save()
    }

    func confirmReset() {
        let alert = NSAlert()
        alert.messageText = "Reset progress?"
        alert.informativeText = "This clears every Known, Review and Read mark."
        alert.addButton(withTitle: "Reset")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        known = []
        review = []
        done = []
        save()
    }

    private func save() {
        defaults.set(Array(known), forKey: "knownCards")
        defaults.set(Array(review), forKey: "reviewCards")
        defaults.set(Array(done), forKey: "readLessons")
    }
}
