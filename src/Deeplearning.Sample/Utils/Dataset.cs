using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Deeplearning.Sample.Utils
{
    public enum IrisType 
    {
        Unknown,
        Setosa,
        Versicolor,
        Virginica,     
    }

    public class IrisData
    {
        public Matrix Iris { get; set; }
        public Matrix Y { get; set; }
        public IrisType[] Types { get; set; }
    }

    public class Dataset
    {

        public static IrisData LoadIrisData(string path, int count = -1,bool random = true)
        {
            string[] lines = File.ReadAllLines(path);

            int dataCount = count > 0 ? count : lines.Length - 1;

            int dimension = 4;

            int startIndex = random ? new Random().Next(0, lines.Length - dataCount) : 0;

            int endIndex = startIndex + dataCount;

            Matrix x = new Matrix(dataCount, dimension);

            Matrix y = new Matrix(4,dataCount);

            IrisType [] types = new IrisType[dataCount];

            for (int i = startIndex, j = 0; i < endIndex; i++, j++)
            {
                string[] words = lines[i + 1].Split(',');

                for (int k = 0; k < dimension; k++)
                {
                    x[j, k] = float.Parse(words[k]);
                }

                string irisType = words[words.Length - 1];

                IrisType type = IrisTypeStringToEnum(irisType);

                 y[(int)type,j] =1;

                types[i] = type;
            }

            return new IrisData() { Iris = x, Y = y,Types = types };// (x,y);

        }

        public static IrisType IrisTypeStringToEnum(string irisType) 
        {
            IrisType type = IrisType.Unknown;

            switch (irisType)
            {
                case "Iris-setosa":
                    type = IrisType.Setosa;
                    break;
                case "Iris-versicolor":
                    type = IrisType.Versicolor;
                    break;
                case "Iris-virginica":
                    type = IrisType.Virginica;
                    break;
                default:
                    break;
            }

            return type;
        }

    }
}
