from rag_pipeline import build_ready_rag_chain, stream_normalized_text


DEFAULT_QUESTIONS = [
    "Kto płaci za zwrot towaru?",
    "Jaki jest termin zwrotu produktu?",
    "Czy moge zwrocic produkt kupiony online?",
    "Jaka jest stolica Niemiec?",
]


def stream_answer(rag_chain, question: str) -> None:
    print("ANSWER: ", end="", flush=True)
    for chunk in stream_normalized_text(rag_chain.stream(question)):
        print(chunk, end="", flush=True)
    print("\n")


def main() -> None:
    print("Preparing RAG. The first run may take a moment...")
    rag_chain = build_ready_rag_chain()

    print("\nDefault questions:")
    for question in DEFAULT_QUESTIONS:
        print(f"\n{'=' * 60}\nQUESTION: {question}")
        stream_answer(rag_chain, question)

    print("Interactive mode. Type a question or exit with: exit, quit, koniec.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", "koniec"}:
            print("Done.")
            break

        if not question:
            continue

        stream_answer(rag_chain, question)


if __name__ == "__main__":
    main()
