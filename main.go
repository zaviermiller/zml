package main

import (
	"fmt"
	"log"
	"time"

	u "github.com/zaviermiller/zml/utils"
	"github.com/zaviermiller/zml/zmlp"
)
func printData(dataSet *u.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)			// print Digit (label)
	u.PrintImage(data.Image)	// print Image
}
func main() {
	dataSet, err := u.ReadTrainSet("mnist")
	if err != nil {
		log.Fatal(err)
	}

	cfg := zmlp.NNConfig {
		InputNeurons: dataSet.W * dataSet.H,
		OutputNeurons: 10,
		HiddenNeurons: 100,
		NumEpochs: 5,
		LearningRate: .3,
	}

	testSet, err := u.ReadTestSet("mnist")
	if err != nil {
		log.Fatal(err)
	}
	testLabels := make([]float64, testSet.N)
	testData := [][]float64{}
	for i, img := range testSet.Data {
		testLabels[i] = float64(img.Digit)
		tmp := []float64 {}
		for _, row := range img.Image {
			for _, val := range row {
				tmp = append(tmp, (float64(val) / 255.0 * 0.99) + 0.01)
			}
		}
		testData = append(testData, tmp)
	}

	var acc int
	for j, img := range testData {
		outputs, err := mlp.Predict(img)
		if err != nil {
			log.Fatal(err)
		}

		best := 0
		highest := 0.0
		for i := 0; i < 10; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target := testLabels[j]
		if float64(best) == target {
			acc++
		} else {
			// fmt.Println(fmt.Sprintf("INCORRECT GUESS %d for number %d", best, target))
		}
	}

	fmt.Print("Accuracy: ")
	fmt.Println(float64(acc) / float64(len(testData)))

	mlp.Save()
	
}