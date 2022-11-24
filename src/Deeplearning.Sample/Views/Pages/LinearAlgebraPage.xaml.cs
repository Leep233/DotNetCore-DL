using Deeplearning.Sample.ViewModels.Pages;
using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Deeplearning.Sample.Views.Pages
{
    /// <summary>
    /// OrthogonalizationPage.xaml 的交互逻辑
    /// </summary>
    public partial class LinearAlgebraPagePage : Page
    {
        private readonly LinearAlgebraPagePageViewModel viewModel;
        public LinearAlgebraPagePage()
        {
            InitializeComponent();

            viewModel = new LinearAlgebraPagePageViewModel();

            this.DataContext = viewModel;   
        }

        private void OnClickPCAButton(object sender, RoutedEventArgs e)
        {
            pcaPanel.Visibility  =pcaPanel.Visibility == Visibility.Visible? Visibility.Collapsed: Visibility.Visible;
        }

        private void OnClickImageCompressionButton(object sender, RoutedEventArgs e)
        {
            imageCompressionPanel.Visibility = imageCompressionPanel.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;
        }
    }
}
