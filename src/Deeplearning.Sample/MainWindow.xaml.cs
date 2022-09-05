using Deeplearning.Sample.Windows;
using OxyPlot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Deeplearning.Sample
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
       private readonly MainWindowViewModel viewModel;

        public static string imgPath = "./Resources/cat01.jpeg";

        public MainWindow()
        {
            InitializeComponent();
            viewModel = new MainWindowViewModel();
            this.DataContext = viewModel;
        }

        private void OnClickSVDDimensionReduction(object sender, RoutedEventArgs e)
        {
            ImageComparisonWindow window = new ImageComparisonWindow(imgPath,1);
            window.Owner = this;
            window.Show();
        }

        private void OnClickPCDDimensionReduction(object sender, RoutedEventArgs e)
        {
            ImageComparisonWindow window = new ImageComparisonWindow(imgPath, 1);
            window.Owner = this;
            window.Show();
        }
    }
}
