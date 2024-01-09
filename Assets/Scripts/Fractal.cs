using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;
using quaternion = Unity.Mathematics.quaternion;
using Random = UnityEngine.Random;

public class Fractal : MonoBehaviour
{
    // FloatMode.Fast rewrites a + b * c to b * c + a
    // due to multiply-add (madd) instruction.
    // performance is better than having a separate add instruction
    //
    // FloatPrecision helps the precision of sin and cos methods, mainly
    // for computing quaternions or basically, trigonometric precisions are
    // lowered to speed things up
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct UpdateFractalLevelJob : IJobFor
    {
        public float scale;
        public float deltaTime;

        // read only to avoid race conditions
        [ReadOnly]
        public NativeArray<FractalPart> parents;
        public NativeArray<FractalPart> parts;

        [WriteOnly]
        public NativeArray<float3x4> matrices;

        public void Execute (int i)
        {
            FractalPart parent = parents[i / 5];
            FractalPart part = parts[i];
            part.spinAngle += part.spinVelocity * deltaTime;

            // upAxis is the axis that points away from its parent
            float3 upAxis = mul(mul(parent.worldRotation, part.rotation), up());
            float3 sagAxis = cross(up(), upAxis);

            float sagMagnitude = length(sagAxis);
            quaternion baseRotation;
            if (sagMagnitude > 0f)
            {
                sagAxis /= sagMagnitude;
                quaternion sagRotation 
                    = quaternion.AxisAngle(sagAxis, part.maxSagAngle * sagMagnitude);
                baseRotation = mul(sagRotation, parent.worldRotation);
            }
            else
            {
                baseRotation = parent.worldRotation;
            }

            part.worldRotation = mul(baseRotation,
                mul(part.rotation, quaternion.RotateY(part.spinAngle))
            );

            part.worldPosition = 
                parent.worldPosition + 
                mul(part.worldRotation, float3(0f, 1.5f * scale, 0f));

            parts[i] = part;

            float3x3 r = float3x3(part.worldRotation) * scale;
            matrices[i] = float3x4(r.c0, r.c1, r.c2, part.worldPosition);
        }
    }

    struct FractalPart
    {
        public float3 worldPosition;
        public Quaternion rotation, worldRotation;
        public float spinAngle, maxSagAngle, spinVelocity;
    }

    static readonly int
        colorAId = Shader.PropertyToID("_ColorA"),
        colorBId = Shader.PropertyToID("_ColorB"),
        matricesId = Shader.PropertyToID("_Matrices"),
        sequenceNumbersId = Shader.PropertyToID("_SequenceNumbers");

    static MaterialPropertyBlock propertyBlock;

    // math library uses radians
    static quaternion[] rotations = {
        quaternion.identity,
        quaternion.RotateZ(-0.5f * PI), quaternion.RotateZ(0.5f * PI),
        quaternion.RotateX(0.5f * PI), quaternion.RotateX(-0.5f * PI)
    };

    NativeArray<FractalPart>[] parts;

    NativeArray<float3x4>[] matrices;

    ComputeBuffer[] matricesBuffers;

    [SerializeField] Mesh mesh, leafMesh;
    [SerializeField] Material material;
    [SerializeField] Gradient gradientA,gradientB;
    [SerializeField] Color leafColorA, leafColorB;
    [SerializeField, Range(0f, 90f)] float maxSagAngleA = 15f, maxSagAngleB = 25f;
    [SerializeField, Range(0f, 90f)] float spinSpeedA = 20f, spinSpeedB = 25f;
    [SerializeField, Range(0f, 1f)] float reverseSpinChance = 0.25f;

    [SerializeField, Range(3, 8)] int depth = 4;

    Vector4[] sequenceNumbers;

    private void OnEnable()
    {
        parts = new NativeArray<FractalPart>[depth];
        matrices = new NativeArray<float3x4>[depth];
        matricesBuffers = new ComputeBuffer[depth];

        // 4x4 matrix has 16 float values. 1 float = 4 bytes, so 16 * 4
        // changed to 12 * 4, making the matrix 3x4 to reduce memory
        // transfer about 25%
        int stride = 12 * 4;

        sequenceNumbers = new Vector4[depth];
        for (int i = 0, length = 1; i < parts.Length; i++, length *= 5)
        {
            // construct the fractal part and the matrix through native arrays
            parts[i] = new NativeArray<FractalPart>(length, Allocator.Persistent);
            matrices[i] = new NativeArray<float3x4>(length, Allocator.Persistent);
            matricesBuffers[i] = new ComputeBuffer(length, stride);

            // for random factor on color
            sequenceNumbers[i] = 
                new Vector4(Random.value, Random.value, Random.value, Random.value);
        }

        parts[0][0] = CreatePart(0);
        // level iteration
        for (int li = 1; li < parts.Length; li++)
        {
            NativeArray<FractalPart> levelParts = parts[li];
            // fractal part iteration
            for (int fpi = 0; fpi < levelParts.Length; fpi += 5)
            {
                // child indices
                for (int ci = 0; ci < 5; ci++)
                {
                    levelParts[fpi + ci] = CreatePart(ci);
                }
            }
        }

        // if (propertyBlock == null) {
        //     propertyBlock = new MaterialPropertyBlock();
        // }
        // or
        propertyBlock ??= new MaterialPropertyBlock();
    }

    private void OnDisable()
    {
        for (int i = 0; i < matricesBuffers.Length; i++)
        {
            matricesBuffers[i].Release();

            // dispose parts and matrices
            parts[i].Dispose();
            matrices[i].Dispose();
        }
        // remove all reference
        parts = null;
        matrices = null;
        matricesBuffers = null;
        sequenceNumbers = null;
    }

    private void OnValidate()
    {
        // when in play mode (fractal parts should be initialized)
        if (parts != null && enabled)
        {
            // resets the fractal to new depth value
            OnDisable();
            OnEnable();
        }
    }

    FractalPart CreatePart (int childIndex) => new FractalPart {
        maxSagAngle = radians(Random.Range(maxSagAngleA, maxSagAngleB)),
        rotation = rotations[childIndex],
        spinVelocity = (Random.value < reverseSpinChance ? -1f : 1f) * 
                        radians(Random.Range(spinSpeedA, spinSpeedB))
    };
   
    private void Update()
    {
        float deltaTime = Time.deltaTime;
        // reference root part
        FractalPart rootPart = parts[0][0];
        // change its rotation
        rootPart.spinAngle += rootPart.spinVelocity * deltaTime;


        /* old ver. Replaced the Euler method and the multiplications involving
         * quaternions
         * 
         * 
         * rootPart.worldRotation = transform.rotation * (rootPart.rotation * 
            Quaternion.Euler(0f, rootPart.spinAngle, 0f));
        */

        // new ver
        rootPart.worldRotation = mul(transform.rotation,
            mul(rootPart.rotation, quaternion.RotateY(rootPart.spinAngle))
        );

        // use transform.position in order to move along all elements with the root element
        rootPart.worldPosition = transform.position;
        // copy back edited rotation on the 2d array
        parts[0][0] = rootPart;

        // no well defined scale, use lossyScale property due to this
        // uses the x component of the scale and ignore any non uniform scale
        float objectScale = transform.lossyScale.x;

        float3x3 r = float3x3(rootPart.worldRotation) * objectScale;
        matrices[0][0] = float3x4(r.c0, r.c1, r.c2, rootPart.worldPosition);

        float scale = objectScale;
        JobHandle jobHandle = default;
        for (int li = 1; li < parts.Length; li++)
        {
            scale *= 0.5f;

            // schedule each part to jobHandle
            jobHandle = new UpdateFractalLevelJob
            {
                deltaTime = deltaTime,
                scale = scale,
                parents = parts[li - 1],
                parts = parts[li],
                matrices = matrices[li]
            }.ScheduleParallel(parts[li].Length, 5, jobHandle);
        }
        // execute jobHandle once all fractals are scheduled
        jobHandle.Complete();

        // use 3 for bounds size. due to the summation value of
        // 2 + (n)summation(i=1) 1/2i
        // the diameter of the fractal goes theretically infinite but > 3 in diameter
        // which means the fractal is guaranteed to fit in a bounding box of 3 units wide
        var bounds = new Bounds(rootPart.worldPosition, 3f * objectScale * Vector3.one);

        int leafIndex = matricesBuffers.Length - 1;
        for (int i = 0; i < matricesBuffers.Length; i++)
        {
            ComputeBuffer buffer = matricesBuffers[i];
            buffer.SetData(matrices[i]);


            Color colorA, colorB;

            Mesh instanceMesh;
            if (i == leafIndex)
            {
                colorA = leafColorA;
                colorB = leafColorB;
                instanceMesh = leafMesh;
            }
            else
            {
                float gradientInterpolator = i / (matricesBuffers.Length - 2f);
                colorA = gradientA.Evaluate(gradientInterpolator);
                colorB = gradientB.Evaluate(gradientInterpolator);
                instanceMesh = mesh;
            }
            propertyBlock.SetColor(colorAId, colorA);
            propertyBlock.SetColor(colorBId, colorB);

            propertyBlock.SetBuffer(matricesId, buffer);
            propertyBlock.SetVector(sequenceNumbersId, sequenceNumbers[i]);

            // draws the same mesh multiple times using GPU instancing
            Graphics.DrawMeshInstancedProcedural(
                instanceMesh, 0, material, bounds, buffer.count, propertyBlock
            );
        }
    }
}



/*
 * Flat Hierarchy Method
 * 
struct FractalPart
{
    public Vector3 direction;
    public Quaternion rotation;
    public Transform transform;
}

FractalPart[][] parts;

[SerializeField] Mesh mesh;
[SerializeField] Material material;

[SerializeField, Range(1, 8)] int depth = 4;

static Vector3[] directions =
{
    Vector3.up, Vector3.right, Vector3.left, Vector3.forward, Vector3.back,
};

static Quaternion[] rotations =
{
    Quaternion.identity,
    Quaternion.Euler(0f, 0f, -90f),
    Quaternion.Euler(0f, 0f, 90f),
    Quaternion.Euler(90f, 0f, 0f),
    Quaternion.Euler(-90f, 0f, 0f),

};

FractalPart CreatePart (int levelIndex, int childIndex, float scale)
{
    var go = new GameObject("Fractal Part L" + levelIndex + " C" + childIndex);
    go.transform.localScale = scale * Vector3.one;
    go.transform.SetParent(transform, false);
    go.AddComponent<MeshFilter>().mesh = mesh;
    go.AddComponent<MeshRenderer>().material = material;

    return new FractalPart {
        direction = directions[childIndex],
        rotation = rotations[childIndex],
        transform = go.transform,
    };
}

private void Awake()
{
    parts = new FractalPart[depth][];

    for (int i = 0, length = 1; i < parts.Length; i++, length *= 5)
    {
        parts[i] = new FractalPart[length];
    }

    float scale = 1f;
    parts[0][0] = CreatePart(0, 0, scale);
    // level iteration
    for (int li = 1; li < parts.Length; li++)
    {
        scale *= 0.5f;
        FractalPart[] levelParts = parts[li];
        // fractal part iteration
        for (int fpi = 0; fpi < levelParts.Length; fpi += 5) { 
            // child indices
            for (int ci = 0; ci < 5; ci++)
            {
                levelParts[fpi + ci] = CreatePart(li, ci, scale);
            }
        }
    }
}

private void Update()
{
    // for animation
    Quaternion deltaRotation = Quaternion.Euler(0f, 22.5f * Time.deltaTime, 0f);

    // reference root part
    FractalPart rootPart = parts[0][0];
    // change its rotation
    rootPart.rotation *= deltaRotation;
    rootPart.transform.localRotation = rootPart.rotation; // makes fractal spin around its root
    // copy back edited rotation on the 2d array
    parts[0][0] = rootPart;

    for (int li = 1; li < parts.Length; li++ )
    {
        FractalPart[] parentParts = parts[li - 1];
        FractalPart[] levelParts = parts[li];
        for (int fpi = 0; fpi < levelParts.Length; fpi++)
        {
            Transform parentTransform = parentParts[fpi / 5].transform;
            FractalPart part = levelParts[fpi];

            part.rotation *= deltaRotation;

            part.transform.localRotation =
                parentTransform.localRotation * part.rotation;

            // use parent of the current level to identify the current part's position
            // relative to its parent. Basically, parent's current position +
            // ( part's scale * part's direction)
            part.transform.localPosition = 
                parentTransform.localPosition + 
                parentTransform.localRotation * 
                (1.5f * part.transform.localScale.x * part.direction);

            levelParts[fpi] = part;

        }
    }
}
*/

/* 
 * Brute Force Method
 * 
 * private void Start()
{
    name = "Fractal" + depth;

    if (depth <= 1)
    {
        return;
    }

    Fractal childA = CreateChild(Vector3.up, Quaternion.identity);
    Fractal childB = CreateChild(Vector3.right, Quaternion.Euler(0f, 0f, -90f));
    Fractal childC = CreateChild(Vector3.left, Quaternion.Euler(0f, 0f, 90f));
    Fractal childD = CreateChild(Vector3.forward, Quaternion.Euler(90f, 0f, 0f));
    Fractal childE = CreateChild(Vector3.back, Quaternion.Euler(-90f, 0f, 0f));

    childA.transform.SetParent(transform, false);
    childB.transform.SetParent(transform, false);
    childC.transform.SetParent(transform, false);
    childD.transform.SetParent(transform, false);
    childE.transform.SetParent(transform, false);

}

void Update()
{
    transform.Rotate(0f, 22.5f * Time.deltaTime, 0f);
}

Fractal CreateChild (Vector3 direction, Quaternion rotation)
{
    Fractal child = Instantiate(this);
    child.depth = depth - 1;
    child.transform.localPosition = 0.75f * direction;
    child.transform.localRotation = rotation;
    child.transform.localScale = 0.5f * Vector3.one;
    return child;
}*/
