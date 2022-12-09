using Deeplearning.Sample.ViewModels.Components;
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

namespace Deeplearning.Sample.Views.Components
{
    /// <summary>
    /// MatrixOperationComponent.xaml 的交互逻辑
    /// </summary>
    public partial class MatrixOperationComponent : UserControl
    {
        private readonly MatrixOperationComponentViewModel viewModel;

        public MatrixOperationComponent()
        {
            viewModel = new MatrixOperationComponentViewModel();

            InitializeComponent();

            DataContext = viewModel;
        }
    }
}
