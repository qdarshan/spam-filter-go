// Naive Bayes Spam Classifier
//
//   P(spam|words) = P(words|spam) * P(spam) / P(words)
//     posterior    =  likelihood   *  prior  / evidence
//
//   prior      — how common spam/ham is overall (spamCount / totalCount)
//   likelihood — probability of seeing these words given it's spam (or ham)
//   evidence   — probability of seeing these words regardless of class
//   posterior   — final score: how likely the email is spam (or ham)
//
//   We use log() on everything so we can add instead of multiply,
//   which avoids floating-point underflow with many small probabilities.

package main

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
)

type Bow map[string]int

const MinWordFreq = 100

func addFileToBow(path string, bow Bow) error {
	content, err := os.ReadFile(path)

	if err != nil {
		return err
	}

	for _, token := range tokenize(string(content)) {
		bow[token] += 1
	}

	return nil
}

func addDirToBow(path string, bow Bow) error {
	return filepath.WalkDir(path, func(path string, d os.DirEntry, err error) error {
		if d.IsDir() {
			return nil
		}

		if err := addFileToBow(path, bow); err != nil {
			return err
		}
		return nil
	})
}

func classifyFile(filepath string, hamBow Bow, hamTotal int, spamBow Bow, spamTotal int) (float64, float64, error) {
	totalCount := hamTotal + spamTotal

	priorHam := float64(hamTotal) / float64(totalCount)
	priorSpam := float64(spamTotal) / float64(totalCount)

	fileBow := make(Bow)
	if err := addFileToBow(filepath, fileBow); err != nil {
		return 0.0, 0.0, err
	}

	logEvidence := 0.0
	logLikelihoodSpam := 0.0
	logLikelihoodHam := 0.0
	for word := range fileBow {

		totalWordFreq := spamBow[word] + hamBow[word]

		if totalWordFreq < MinWordFreq {
			continue
		}

		if spamBow[word] != 0 {
			logLikelihoodSpam += math.Log(float64(spamBow[word]) / float64(spamTotal))
		}

		if hamBow[word] != 0 {
			logLikelihoodHam += math.Log(float64(hamBow[word]) / float64(hamTotal))
		}

		if totalWordFreq != 0 {
			logEvidence += math.Log(float64(totalWordFreq) / float64(totalCount))
		}
	}

	spamScore := logLikelihoodSpam + priorSpam - logEvidence
	hamScore := logLikelihoodHam + priorHam - logEvidence

	return spamScore, hamScore, nil
}

func classifyDir(dirPath string, hamBow Bow, hamTotal int, spamBow Bow, spamTotal int) (int, int, error) {
	spamCount := 0
	hamCount := 0
	filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {

		if d.IsDir() {
			return nil
		}

		spamScore, hamScore, err := classifyFile(path, hamBow, hamTotal, spamBow, spamTotal)

		if err != nil {
			return err
		}

		if spamScore > hamScore {
			spamCount++
		} else {
			hamCount++
		}
		return nil
	})
	return spamCount, hamCount, nil
}

func tokenize(message string) []string {
	tokens := strings.Fields(message)
	for i := range tokens {
		tokens[i] = strings.ToUpper(tokens[i])
	}
	return tokens
}

func totalWordCount(bow Bow) int {
	count := 0
	for word := range bow {
		if bow[word] < MinWordFreq {
			continue
		}
		count += bow[word]
	}
	return count
}

func main() {
	hamBow := make(Bow)
	spamBow := make(Bow)

	fmt.Println(">> training <<")
	for i := 1; i <= 5; i++ {
		err := addDirToBow(fmt.Sprintf("data/enron%v/ham", i), hamBow)
		if err != nil {
			panic(err)
		}

		err = addDirToBow(fmt.Sprintf("data/enron%v/spam", i), spamBow)
		if err != nil {
			panic(err)
		}
	}

	hamTotal := totalWordCount(hamBow)
	spamTotal := totalWordCount(spamBow)

	fmt.Println(">> classify ham <<")
	spamCount, hamCount, err := classifyDir("data/enron6/ham", hamBow, hamTotal, spamBow, spamTotal)
	fmt.Printf("spam: %d \n ham: %d \n", spamCount, hamCount)
	if err != nil {
		panic(err)
	}

	fmt.Println(">> classify spam <<")
	spamCount, hamCount, err = classifyDir("data/enron6/spam", hamBow, hamTotal, spamBow, spamTotal)
	fmt.Printf("spam: %d \n ham: %d \n", spamCount, hamCount)
	if err != nil {
		panic(err)
	}
}
