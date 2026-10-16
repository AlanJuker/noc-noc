# Benchmark

## Task

Given a job title, return the NOC 2021 unit group it belongs to. That's what an applicant has in hand.

## Data

Statistics Canada lists 27,935 example job titles, each under exactly one of the 516 unit groups ("All
examples" in the NOC 2021 V1.0 elements file).

## Split

A title goes to the test set when the SHA-256 of its text is divisible by 5:

- **5,468 test titles** covering 506 unit groups; 22,467 stay in the index
- The split depends only on the text: no random seed, same result on every machine
- Test titles are removed from the index, so no method can find a test title by looking it up
- Every unit group keeps at least one example title in the index

Near-duplicates stay on both sides on purpose (`web programmer` held out, `web developer` indexed), because
real users type titles that resemble the official ones without matching them.

## Metrics

| Metric | Meaning |
|---|---|
| top-1 | the first answer is right |
| top-5 | the right code is among the first five, the list a user actually reads |
| MRR | mean reciprocal rank of the right code within the top 10 |
| confusable error share | of the wrong first answers, the share that Statistics Canada's own exclusions list as easy to confuse with the right one |

The last one separates "the method is bad" from "the classification itself is ambiguous here".
