using Deeplearning.Sample.Utils;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace Deeplearning.Sample
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {

        const string irisTrainingPath = "./resources/testdata/iris_training.csv";
        const string irisTestingPath = "./resources/testdata/iris_testing.csv";

        private static IrisData IrisTrainData;

        private static IrisData IrisTestData;

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
        }

        public static IrisData GetIrisTrainData() { 
            if(IrisTrainData is null)
                IrisTrainData = IrisHelper.LoadIrisData(irisTrainingPath);
            return IrisTrainData;
        }

        public static IrisData GetIrisTestData()
        {
            if (IrisTestData is null)
                IrisTestData = IrisHelper.LoadIrisData(irisTestingPath);
            return IrisTestData;
        }
    }
}
