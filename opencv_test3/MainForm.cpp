#include "MainForm.h"
using namespace System;
using namespace System::Windows::Forms;
[STAThreadAttribute]
//void Main(array<String^> ^args) {
	int main(int, char**){
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	opencv_test3::MainForm form;
	Application::Run(%form);
}