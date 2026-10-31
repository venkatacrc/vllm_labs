import Foundation

/// Mirrors web/course.json, written by tools/build_course.py.
struct Course: Decodable {
    let title: String
    let repo: String
    let version: String
    let tracks: [Track]

    static let empty = Course(title: "LLM Serving Labs", repo: "", version: "", tracks: [])
}

struct Track: Decodable, Identifiable {
    let id: String
    let title: String
    let blurb: String
    let lessons: [Lesson]
}

struct Lesson: Decodable, Identifiable, Hashable {
    let id: String
    let title: String
    let kind: String
    let source: String
    let url: String
    let html: String
    let cards: [Card]

    static func == (a: Lesson, b: Lesson) -> Bool { a.id == b.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct Card: Decodable, Identifiable {
    let id: String
    let q: String
    let a: String
    let qHtml: String
    let aHtml: String
}

/// A card together with the lesson it belongs to.
struct StudyCard: Identifiable {
    let card: Card
    let lesson: Lesson
    var id: String { card.id }
}

enum Mode: String { case read, cards }
enum Study: String { case learn, test }
enum Scope: String, CaseIterable { case lesson, track, all }
enum Practice: String, CaseIterable { case all, notKnown, review }
