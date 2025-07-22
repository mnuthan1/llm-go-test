package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

type FileData struct {
	Path   string                 `json:"path"`
	Values map[string]interface{} `json:"values"`
}

type ConfigTree struct {
	Chart   string     `json:"chart"`
	Configs []FileData `json:"configs"`
}

type ExpectedWarning struct {
	Path    string `json:"path"`
	Key     string `json:"key"`
	Message string `json:"message"`
}

type TestCase struct {
	Name             string            `json:"name"`
	Input            ConfigTree        `json:"input"`
	ExpectedWarnings []ExpectedWarning `json:"expected_warnings"`
}

func TestLinterAccuracy(t *testing.T) {
	files, err := os.ReadDir("test_cases")
	if err != nil {
		t.Fatalf("Failed to read test_cases directory: %v", err)
	}

	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".json") {
			continue
		}
		t.Run(file.Name(), func(t *testing.T) {
			path := "test_cases/" + file.Name()
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("Failed to read file %s: %v", path, err)
			}

			var tc TestCase
			if err := json.Unmarshal(data, &tc); err != nil {
				t.Fatalf("Failed to unmarshal test case: %v", err)
			}

			prompt := fmt.Sprintf(`You are a YAML configuration linter that analyzes hierarchical configuration trees.

			Configuration Hierarchy & Overrides:
			Configuration files are organized hierarchically, following a structure like chart1/falcon/env/dev/values.yaml.
			Chart Base: The first path segment (e.g., chart1/) defines the chart base, files in base folder are not overrides.
			Override Layers: Any subfolders named falcon or deeper within a path (e.g., chart1/falcon/...) represent override layers.
			Parent-Child Relationship: The file path hierarchy dictates parent-child relationships for override detection. For instance, chart1/values.yaml is the parent of chart1/falcon/env/dev/values.yaml. 
			
			Linter Rules:
			1. Identify and report the following issues:
			2. Duplicate Keys (Same Level): A key is defined in multiple files at the same hierarchical level (e.g., chart1/values.yaml and chart1/default.yaml).
			3. Redundant Override: An override file (within a falcon layer) sets a key to the exact same value as its parent configuration file. (Note: Differing values are valid overrides and should not be flagged).
			4. Override-Only Key: A key is introduced only within an override layer (falcon/...) and does not exist in its direct parent configuration file.
			5. Hardcoded Sensitive Values: The configuration contains values matching patterns for sensitive data:
			   - AWS regions (e.g., us-west-1, ap-southeast-2)
			   - Account IDs (12-digit numbers)
			   - ARNs (starting with arn:)
			   - Common secret identifiers (e.g., key, token, password, credential) 
			
			Output Format:
			For each detected issue, provide:
			- File Path
			- Key
			- Value
			- Warning Type & Suggestion
			
			Constraints:
			Analyze only the provided configuration data. Do not infer or invent keys, values, or file paths.
			Generate warnings only for keys explicitly present in the input. 
			
			Now analyze this configuration tree:
			
			%s`, marshalTree(tc.Input))

			reqBody := map[string]interface{}{
				"model":       "mistral",
				"prompt":      prompt,
				"stream":      false,
				"temperature": 0,
			}

			reqBytes, _ := json.Marshal(reqBody)
			resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(reqBytes))
			if err != nil {
				t.Fatalf("Failed to call LLM: %v", err)
			}
			defer resp.Body.Close()

			var result map[string]interface{}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				t.Fatalf("Failed to parse LLM response: %v", err)
			}

			text := result["response"].(string)
			extracted := extractWarnings(text)
			precision, recall := scoreWarnings(tc.ExpectedWarnings, extracted)
			t.Logf("Precision = %.2f, Recall = %.2f", precision, recall)

			if precision < 1.0 || recall < 1.0 {
				t.Errorf("Failed test case %s: extracted = %+v", tc.Name, extracted)
			}
		})
	}
}

func marshalTree(tree ConfigTree) string {
	bytes, _ := json.MarshalIndent(tree, "", "  ")
	return string(bytes)
}

func extractWarnings(text string) []ExpectedWarning {
	var warnings []ExpectedWarning
	pattern := regexp.MustCompile(`(?m)- File Path: (.*?), Key: (.*?), Value: .*?\n.*?Suggestion: (.*?)\n`)
	matches := pattern.FindAllStringSubmatch(text, -1)
	for _, m := range matches {
		w := ExpectedWarning{
			Path:    strings.TrimSpace(m[1]),
			Key:     strings.TrimSpace(m[2]),
			Message: strings.TrimSpace(m[3]),
		}
		warnings = append(warnings, w)
	}
	return warnings
}

func normalizePath(path string) []string {
	parts := strings.Split(path, ",")
	var trimmed []string
	for _, p := range parts {
		trimmed = append(trimmed, strings.TrimSpace(p))
	}
	sort.Strings(trimmed)
	return trimmed
}

func hasCommonElement(arr1, arr2 []string) bool {
	// Create a map to store elements from the first array for efficient lookups.
	elements := make(map[string]bool)
	for _, val := range arr1 {
		elements[val] = true
	}

	// Iterate through the second array and check if any element exists in the map.
	for _, val := range arr2 {
		if elements[val] {
			return true // Found a common element
		}
	}
	return false // No common element found
}

func comparePaths(p1, p2 string) bool {
	n1 := normalizePath(p1)
	n2 := normalizePath(p2)
	return hasCommonElement(n1, n2)
}

func cosineSimilarity(a, b string) float64 {
	awords := strings.Fields(strings.ToLower(a))
	bwords := strings.Fields(strings.ToLower(b))
	wordSet := map[string]bool{}
	for _, w := range awords {
		wordSet[w] = true
	}
	for _, w := range bwords {
		wordSet[w] = true
	}
	vecA := make([]float64, 0, len(wordSet))
	vecB := make([]float64, 0, len(wordSet))
	for w := range wordSet {
		vecA = append(vecA, float64(count(awords, w)))
		vecB = append(vecB, float64(count(bwords, w)))
	}
	return dot(vecA, vecB) / (magnitude(vecA)*magnitude(vecB) + 1e-8)
}

func count(words []string, target string) int {
	c := 0
	for _, w := range words {
		if w == target {
			c++
		}
	}
	return c
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func magnitude(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func scoreWarnings(expected, actual []ExpectedWarning) (precision float64, recall float64) {
	match := 0
	used := make([]bool, len(actual))
	for _, e := range expected {
		for i, a := range actual {
			if used[i] {
				continue
			}
			pathScore := 0.0
			keyScore := 0.0
			msgScore := cosineSimilarity(e.Message, a.Message)

			if comparePaths(e.Path, a.Path) {
				pathScore = 1.0
			}
			if e.Key == a.Key {
				keyScore = 1.0
			}

			totalScore := 0.4*pathScore + 0.4*keyScore + 0.2*msgScore
			if totalScore >= 0.75 {
				match++
				used[i] = true
				break
			}
		}
	}

	if len(actual) > 0 {
		precision = float64(match) / float64(len(actual))
	} else if len(expected) == 0 {
		precision = 1.0
	} else {
		precision = 0.0
	}

	if len(expected) > 0 {
		recall = float64(match) / float64(len(expected))
	} else {
		recall = 1.0
	}
	return
}
