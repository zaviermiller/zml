package main

import (
	"fmt"
	"log"
	"time"
	"sync"

	u "github.com/zaviermiller/zml/utils"
	// "github.com/zaviermiller/zml/znn"
	"github.com/zaviermiller/zml/zdnn"
)
func printData(dataSet *u.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)			// print Digit (label)
	u.PrintImage(data.Image)	// print Image
}
func main() {
	dataSet, err := u.ReadTrainSet("data")
	if err != nil {
		log.Fatal(err)
	}

	// set up basic neural net
	
	// cfg := znn.NNConfig {
	// 	InputNeurons: dataSet.W * dataSet.H,
	// 	OutputNeurons: 10,
	// 	HiddenNeurons: 100,
	// 	NumEpochs: 5,
	// 	LearningRate: .1,
	// }

	// mlp := znn.NewNetwork(cfg)

	// set up deep neural net
	layercfg := zdnn.LayerConfig { Neurons: 100 }
	layers := []*zdnn.NeuronLayer { zdnn.NewLayer(layercfg), zdnn.NewLayer(layercfg) }
	dcfg := zdnn.NNConfig {
		InputNeurons: dataSet.W * dataSet.H,
		OutputNeurons: 10,
		HiddenLayers: layers,
		NumEpochs: 10,
		LearningRate: .1,
	}

	dnn := zdnn.NewNetwork(dcfg)


	// format data
	digitsData := make([][]float64, dataSet.N)
	inputsData := [][]float64{}
	for i, img := range dataSet.Data {
		digArr := make([]float64, 10)
		for i, _ := range digArr {
			if (img.Digit) == i {
				digArr[i] = 1.0
				continue
			}
			digArr[i] = 0.0
		}
		digitsData[i] = digArr
		tmp := []float64{}
		for _, row := range img.Image {
			for _, val := range row {
				tmp = append(tmp, (float64(val) / 255.0 * 1.0))
			}
		}
		inputsData = append(inputsData, tmp)
	}

	t1 := time.Now()
	fmt.Println("Beginning to train...")

	var wg sync.WaitGroup

	for e := 0; e < dcfg.NumEpochs; e++ {
		wg.Add(1)
		go dnn.Train(inputsData, digitsData, dataSet.N, &wg)
	}

	wg.Wait()

	fmt.Println(fmt.Sprintf("done in %s! testing...", time.Since(t1)))

	testSet, err := u.ReadTestSet("data")
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
				tmp = append(tmp, (float64(val) / 255.0))
			}
		}
		testData = append(testData, tmp)
	}

	var acc int
	for j, img := range testData {
		outputs, err := dnn.Predict(img)
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

	// mlp.Save()
	
}