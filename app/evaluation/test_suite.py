"""
3GPP RAG Evaluation Test Suite

Contains test cases designed to validate:
1. Faithfulness — Are answers grounded in source documents?
2. Refusal — Does the system refuse when context is insufficient?
3. Adversarial — Can trick questions cause hallucination?
4. Accuracy — Are 3GPP-specific answers technically correct?

Usage:
    python -m app.evaluation.run_eval
"""

# Each test case has:
# - question: The user query
# - expected_behavior: What the system should do
# - category: Type of test
# - keywords_expected: Key terms that should appear in a correct answer (if applicable)
# - should_refuse: Whether the system should refuse to answer

EVALUATION_TEST_CASES = [
    # ═══════════════════════════════════════════════════════════
    # CATEGORY 1: FAITHFULNESS TESTS
    # These questions should be answerable from 3GPP docs.
    # The system should cite sources and give grounded answers.
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What is the role of the AMF in 5G network architecture?",
        "expected_behavior": "Should describe AMF functions from TS 23.501 with citations",
        "category": "faithfulness",
        "keywords_expected": ["access", "mobility", "management", "registration", "NAS"],
        "should_refuse": False,
    },
    {
        "question": "Describe the registration procedure in 5G as defined in the 3GPP specifications.",
        "expected_behavior": "Should describe the registration flow from TS 23.502 with step-by-step procedure",
        "category": "faithfulness",
        "keywords_expected": ["registration", "UE", "AMF", "authentication"],
        "should_refuse": False,
    },
    {
        "question": "What is the difference between N1, N2, and N3 reference points?",
        "expected_behavior": "Should explain each reference point with its endpoints",
        "category": "faithfulness",
        "keywords_expected": ["N1", "N2", "N3", "UE", "AMF", "UPF"],
        "should_refuse": False,
    },
    {
        "question": "Explain the PDU session establishment procedure.",
        "expected_behavior": "Should describe the PDU session establishment from TS 23.502",
        "category": "faithfulness",
        "keywords_expected": ["PDU", "session", "SMF", "UPF"],
        "should_refuse": False,
    },
    {
        "question": "What is network slicing and how does it work in 5G?",
        "expected_behavior": "Should explain S-NSSAI, NSI, and slice selection",
        "category": "faithfulness",
        "keywords_expected": ["slice", "S-NSSAI", "NSSF"],
        "should_refuse": False,
    },

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 2: REFUSAL TESTS
    # These questions are OUTSIDE the scope of 3GPP docs.
    # The system SHOULD refuse to answer.
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What is the capital of France?",
        "expected_behavior": "Should refuse — this is not in 3GPP documents",
        "category": "refusal",
        "keywords_expected": [],
        "should_refuse": True,
    },
    {
        "question": "Write me a Python function to sort a list.",
        "expected_behavior": "Should refuse — not related to telecom",
        "category": "refusal",
        "keywords_expected": [],
        "should_refuse": True,
    },
    {
        "question": "What will 6G look like in 2035?",
        "expected_behavior": "Should refuse — speculative, not in 3GPP standards",
        "category": "refusal",
        "keywords_expected": [],
        "should_refuse": True,
    },

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 3: ADVERSARIAL TESTS
    # Trick questions designed to elicit hallucination.
    # ═══════════════════════════════════════════════════════════
    {
        "question": "Explain the role of the QMF (Quantum Management Function) in 5G.",
        "expected_behavior": "Should refuse or state QMF doesn't exist — it's a made-up term",
        "category": "adversarial",
        "keywords_expected": [],
        "should_refuse": True,
    },
    {
        "question": "What is the N99 reference point used for in 3GPP Release 18?",
        "expected_behavior": "Should refuse — N99 doesn't exist",
        "category": "adversarial",
        "keywords_expected": [],
        "should_refuse": True,
    },
    {
        "question": "Describe how the UPF handles AI-based traffic optimization as specified in TS 23.501.",
        "expected_behavior": "Should refuse or clarify — UPF doesn't do AI-based optimization in the spec",
        "category": "adversarial",
        "keywords_expected": [],
        "should_refuse": True,
    },
    {
        "question": "In 3GPP TS 23.501, what is the maximum number of PDU sessions a UE can maintain simultaneously?",
        "expected_behavior": "Should refuse if exact number isn't in context, or answer with citation if it is",
        "category": "adversarial",
        "keywords_expected": [],
        "should_refuse": True,  # Unlikely to be in the document — tests if it makes up a number
    },

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 4: TECHNICAL ACCURACY TESTS
    # Specific 3GPP questions that require precise answers.
    # ═══════════════════════════════════════════════════════════
    {
        "question": "What are the key functions of the SMF?",
        "expected_behavior": "Should list SMF functions: session management, IP allocation, UPF selection, etc.",
        "category": "accuracy",
        "keywords_expected": ["session", "management", "IP", "UPF"],
        "should_refuse": False,
    },
    {
        "question": "What is the purpose of the NRF in the 5G service-based architecture?",
        "expected_behavior": "Should describe NRF as service discovery and registration function",
        "category": "accuracy",
        "keywords_expected": ["NRF", "discovery", "registration", "service"],
        "should_refuse": False,
    },
    {
        "question": "Explain the difference between SUPI and SUCI.",
        "expected_behavior": "Should explain SUPI as permanent ID and SUCI as concealed ID",
        "category": "accuracy",
        "keywords_expected": ["SUPI", "SUCI", "permanent", "concealed"],
        "should_refuse": False,
    },
    {
        "question": "What is the role of the AUSF in the 5G authentication process?",
        "expected_behavior": "Should describe AUSF handling authentication procedures",
        "category": "accuracy",
        "keywords_expected": ["AUSF", "authentication"],
        "should_refuse": False,
    },
    {
        "question": "Describe the service-based architecture (SBA) in 5G core network.",
        "expected_behavior": "Should describe SBA with NFs communicating via service-based interfaces",
        "category": "accuracy",
        "keywords_expected": ["service", "based", "architecture", "NF"],
        "should_refuse": False,
    },
]
