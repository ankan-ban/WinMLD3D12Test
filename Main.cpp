// Minimal C++ example for using WinML with D3D12.
// 
// Goals:
//   - Avoid CPU <-> GPU transfers at each inference (TODO: also demonstrate how to use d3d12 copy queue to pipeline the copies)
//   - pipeline multiple inference requests to keep GPU occupied all the time.

const int warmupIterations = 100;
const int iterations = 100;

#include <stdio.h>
#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Microsoft.AI.MachineLearning.h>
#include <microsoft.ai.machinelearning.native.h>


using namespace winrt::Microsoft::AI::MachineLearning;

#include "Common.h"
#include <chrono>

int main()
{
    HRESULT hr = S_OK;

    ID3D12Device* pDevice;
    ID3D12CommandQueue* pCommandQueue;
    ID3D12Resource* pInput;
    ID3D12Resource* pOutput;

    // 1. Create Device
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

    // 2. Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));

    // 3. Create d3d12 resources (to be used for input and output of the network)
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pInput);
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pOutput);
    uploadInputImageToD3DResource(pDevice, pCommandQueue, pInput, "input.png");

    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFence = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

    // 4. create LearningModelDevice from command queue	
    winrt::com_ptr<ILearningModelDeviceFactoryNative> dFactory =
        winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
    winrt::com_ptr<::IUnknown> spLearningDevice;
    hr = dFactory->CreateFromD3D12CommandQueue(pCommandQueue, spLearningDevice.put());
    LearningModelDevice pWinMLDevice = spLearningDevice.as<LearningModelDevice>();

    // 5. Load the onnx model from file
    auto model = LearningModel::LoadFromFilePath(L"fns-candy.onnx");    // fns-candy

    LearningModelSessionOptions options = {};
    // Important - always specify/override all named dimensions
    // (By default they are set to 1, but DML optimizations get turned off if a model
    auto name = L"None";
    options.OverrideNamedDimension(name, (uint32_t)1);
    options.CloseModelOnSessionCreation(true);

    LearningModelSession session(model, LearningModelDevice(pWinMLDevice), options);
    LearningModelBinding binding(session);

    // 6. Create WinML tensor Objects out of d3d12 resources and bind them to the model.
    TensorFloat inputTensor(nullptr), outputTensor(nullptr);

    winrt::com_ptr<ITensorStaticsNative> tensorfactory = winrt::get_activation_factory<TensorFloat, ITensorStaticsNative>();
    winrt::com_ptr<::IUnknown> spUnkTensor;
    int64_t shapes[4] = { 1, 3, 720, 720 };
    hr = tensorfactory->CreateFromD3D12Resource(pInput, shapes, 4, spUnkTensor.put());
    spUnkTensor.try_as(inputTensor);
    hr = tensorfactory->CreateFromD3D12Resource(pOutput, shapes, 4, spUnkTensor.put());
    spUnkTensor.try_as(outputTensor);

    // Use *Undocumented* property 'DisableTensorCpuSync' to avoid copying back data from GPU
    winrt::Windows::Foundation::Collections::PropertySet bindProperties;
    bindProperties.Insert(L"DisableTensorCpuSync", winrt::Windows::Foundation::PropertyValue::CreateBoolean(true));
    binding.Bind(model.InputFeatures().GetAt(0).Name(), inputTensor, bindProperties);

    bindProperties.Insert(L"DisableTensorCpuSync", winrt::Windows::Foundation::PropertyValue::CreateBoolean(true));
    binding.Bind(model.OutputFeatures().GetAt(0).Name(), outputTensor, bindProperties);


    // 7. Run the model (schedule 100 iterations on the command queue for testing)
    // 
    // Warmup
    for (int i = 1; i <= warmupIterations; i++)
    {
        session.EvaluateAsync(binding, L"");
        pCommandQueue->Signal(pFence, i);
        pFence->SetEventOnCompletion(i, hEvent);    // immediately wait for the GPU results
        DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
    }
	

    // Actual run for benchmarking
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        session.EvaluateAsync(binding, L"");
        pCommandQueue->Signal(pFence, i + 1);

        // wait for (i-2)nd iteration (so that we have 2 iterations in flight)
        if (i > 1)
        {
            pFence->SetEventOnCompletion(i - 1, hEvent);
            DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
        }
    }

    // Wait for the last iteration
    pFence->SetEventOnCompletion(iterations, hEvent);
    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // save the output to disk
    saveOutputImageFromD3DResource(pDevice, pCommandQueue, pOutput, "output.png");
	
    // 8. Release d3d12 objects
    pFence->Release();
    pInput->Release();
    pOutput->Release();
    pCommandQueue->Release();
    pDevice->Release();

    printf("\nInference loop done. %d iterations in %g ms - avg: %g ms per iteration\n", iterations, duration, duration/iterations);
    return 0;
}