package main

import (
	"fmt"
	"log"
	"time"

	// "sync"

	u "github.com/zaviermiller/zml/utils"
	// "github.com/zaviermiller/zml/znn"
	"github.com/zaviermiller/zml/zdnn"
)

func printData(dataSet *u.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)  // print Digit (label)
	u.PrintImage(data.Image) // print Image
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

	// set up deep neural nets layers and config
	layercfg := zdnn.LayerConfig{Neurons: 20, Activation: zdnn.Sigmoid}
	outputLayer := zdnn.NewLayer(zdnn.LayerConfig{Neurons: 10, Activation: zdnn.Sigmoid})
	layers := []*zdnn.NeuronLayer{zdnn.NewLayer(layercfg), zdnn.NewLayer(layercfg)}
	dcfg := zdnn.NNConfig{
		InputNeurons: dataSet.H * dataSet.W,
		HiddenLayers: layers,
		OutputLayer:  outputLayer,
		NumEpochs:    10,
		LearningRate: .3,
		LossFunc:     zdnn.MeanSquared,
		BatchSize:    20,
	}

	// build the network
	dnn := zdnn.NewNetwork(dcfg)

	// format data
	digitsData := make([][]float64, dataSet.N)
	inputsData := [][]float64{}
	for i, img := range dataSet.Data {
		digArr := make([]float64, 10)
		for i := range digArr {
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

	dnn.Train(inputsData, digitsData, dataSet.N)

	fmt.Println(fmt.Sprintf("done in %s! testing...", time.Since(t1)))

	testSet, err := u.ReadTestSet("data")
	if err != nil {
		log.Fatal(err)
	}
	testLabels := make([]float64, testSet.N)
	testData := [][]float64{}
	for i, img := range testSet.Data {
		testLabels[i] = float64(img.Digit)
		tmp := []float64{}
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
