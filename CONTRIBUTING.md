# CONTRIBUTION GUIDELINE

## Prerequisites
* Python >=3.9

## Contribution
1. Fork the library and clone to your local machine
2. Install library and dependencies using
   * ```make develop # Download all prerequisites ```
   * ```make build # Build the library```
   * ```make install # Install the gym environment```

## Making a PR
1. Run ```make lint``` and ```make test``` and make sure all test passed
2. Submit the PR

## What to focus on now?
* More substantial test (Coverage is still low for now)
* Decoupling Personality.py so that adding new Personality would be easier
* Adding ways to use custom reward function except ones given