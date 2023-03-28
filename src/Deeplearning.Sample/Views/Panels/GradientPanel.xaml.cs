using Deeplearning.Sample.ViewModels.Panels;
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

namespace Deeplearning.Sample.Views.Panels
{
    /// <summary>
    /// GradientPanel.xaml 的交互逻辑
    /// </summary>
    public partial class GradientPanel : UserControl
    {

        private readonly GradientPanelViewModel viewModel;
        public GradientPanel()
        {
            viewModel = new GradientPanelViewModel();
            InitializeComponent();
            this.DataContext = viewModel;   
        }
    }
}
