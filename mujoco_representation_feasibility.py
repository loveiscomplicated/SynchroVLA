from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from vla_gnn_recurrent.training.representation_feasibility import (
    KeypointDiagnosticConfig,
    KeypointExperimentConfig,
    KeypointStructuralDiagnosticConfig,
    PartNodeDemoConfig,
    PartNodeEvalConfig,
    PartNodePairedEvalConfig,
    PartNodeTrainConfig,
    RepresentationDemoConfig,
    RepresentationEvalConfig,
    RepresentationTrainConfig,
    evaluate_part_node_paired_rollouts,
    evaluate_part_node_policy,
    evaluate_representation_policy,
    generate_part_node_demonstrations,
    generate_representation_demonstrations,
    run_keypoint_diagnostics,
    run_keypoint_feasibility_experiment,
    run_keypoint_structural_diagnostics,
    train_part_node_policy,
    train_representation_policy,
)
from vla_gnn_recurrent.training.generalized_geometric_graph import (
    GeneralizedGeometryConfig,
    OrientationSanityConfig,
    run_generalized_experiment,
    run_orientation_sanity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MuJoCo representation feasibility ablation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("generate", help="Generate shared handle-contact demonstrations.")
    demo.add_argument("--episodes", type=int, default=120)
    demo.add_argument("--max-steps", type=int, default=24)
    demo.add_argument("--seed", type=int, default=2401)
    demo.add_argument("--output-path", default="artifacts/representation_feasibility/demos/handle_contact.pt")
    demo.add_argument("--metadata-path", default="artifacts/representation_feasibility/demos/handle_contact_metadata.json")
    demo.add_argument("--object-x-range", nargs=2, type=float, default=(-0.14, 0.14))
    demo.add_argument("--object-z-range", nargs=2, type=float, default=(0.50, 0.74))
    demo.add_argument("--object-yaw-range", nargs=2, type=float, default=(-0.9, 0.9))
    demo.add_argument("--handle-offset-range", nargs=2, type=float, default=(0.075, 0.115))
    demo.add_argument("--target-radius", type=float, default=0.022)
    demo.add_argument("--max-delta-ee", type=float, default=0.035)
    demo.add_argument("--camera-yaw", type=float, default=0.0)
    demo.add_argument("--crop-size", type=int, default=32)
    demo.add_argument("--visual-grid", type=int, default=8)
    demo.add_argument("--render-first-episode", action="store_true")

    train = subparsers.add_parser("train", help="Train one A/B/C representation variant.")
    train.add_argument("--dataset-path", default="artifacts/representation_feasibility/demos/handle_contact.pt")
    train.add_argument("--variant", choices=["pose", "pose_visual", "pose_visual_ee"], default="pose")
    train.add_argument("--model-kind", choices=["mlp", "mpnn"], default="mlp")
    train.add_argument("--output-dir", default="artifacts/representation_feasibility/checkpoints")
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--mpnn-hidden-dim", type=int, default=64)
    train.add_argument("--mpnn-layers", type=int, default=2)
    train.add_argument("--seed", type=int, default=2401)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--log-every", type=int, default=10)

    eval_parser = subparsers.add_parser("eval", help="Closed-loop rollout evaluation for one variant.")
    eval_parser.add_argument("--checkpoint-path", required=True)
    eval_parser.add_argument("--episodes", type=int, default=48)
    eval_parser.add_argument("--max-steps", type=int, default=24)
    eval_parser.add_argument("--seed", type=int, default=3401)
    eval_parser.add_argument("--output-dir", default="artifacts/representation_feasibility/eval")
    eval_parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    eval_parser.add_argument("--camera-yaw", type=float, default=None)
    eval_parser.add_argument("--render-first-episode", action="store_true")

    compare = subparsers.add_parser("compare", help="Summarize trained checkpoints and eval JSON files.")
    compare.add_argument("--checkpoint-dir", default="artifacts/representation_feasibility/checkpoints")
    compare.add_argument("--eval-dir", default="artifacts/representation_feasibility/eval")

    arch = subparsers.add_parser("compare-architecture", help="Summarize paired C+MLP vs C+MPNN results.")
    arch.add_argument("--mlp-summaries", nargs="+", required=True)
    arch.add_argument("--mpnn-summaries", nargs="+", required=True)
    arch.add_argument("--mlp-evals", nargs="+", required=True)
    arch.add_argument("--mpnn-evals", nargs="+", required=True)
    arch.add_argument("--output-path", default=None)

    part_demo = subparsers.add_parser("part-generate", help="Generate oracle object-part reaching demonstrations.")
    part_demo.add_argument("--episodes", type=int, default=180)
    part_demo.add_argument("--ood-episodes", type=int, default=72)
    part_demo.add_argument("--max-steps", type=int, default=24)
    part_demo.add_argument("--seed", type=int, default=2601)
    part_demo.add_argument("--output-path", default="artifacts/part_node_feasibility/demos/oracle_part_reach.pt")
    part_demo.add_argument(
        "--metadata-path",
        default="artifacts/part_node_feasibility/demos/oracle_part_reach_metadata.json",
    )
    part_demo.add_argument("--object-x-range", nargs=2, type=float, default=(-0.14, 0.14))
    part_demo.add_argument("--object-z-range", nargs=2, type=float, default=(0.50, 0.74))
    part_demo.add_argument("--object-yaw-range", nargs=2, type=float, default=(-0.9, 0.9))
    part_demo.add_argument("--train-part-local-x-range", nargs=2, type=float, default=(-0.035, 0.035))
    part_demo.add_argument("--train-part-local-z-range", nargs=2, type=float, default=(-0.014, 0.026))
    part_demo.add_argument("--ood-part-local-abs-x-range", nargs=2, type=float, default=(0.075, 0.120))
    part_demo.add_argument("--ood-part-local-z-range", nargs=2, type=float, default=(-0.052, 0.052))
    part_demo.add_argument("--target-radius", type=float, default=0.022)
    part_demo.add_argument("--max-delta-ee", type=float, default=0.035)
    part_demo.add_argument("--camera-yaw", type=float, default=0.0)
    part_demo.add_argument("--crop-size", type=int, default=32)
    part_demo.add_argument("--visual-grid", type=int, default=8)
    part_demo.add_argument("--render-first-episode", action="store_true")

    part_train = subparsers.add_parser("part-train", help="Train one oracle part-node ablation variant.")
    part_train.add_argument("--dataset-path", default="artifacts/part_node_feasibility/demos/oracle_part_reach.pt")
    part_train.add_argument("--variant", choices=["object_only", "part_concat", "part_node"], default="object_only")
    part_train.add_argument("--output-dir", default="artifacts/part_node_feasibility/checkpoints")
    part_train.add_argument("--epochs", type=int, default=36)
    part_train.add_argument("--batch-size", type=int, default=256)
    part_train.add_argument("--learning-rate", type=float, default=3e-4)
    part_train.add_argument("--weight-decay", type=float, default=1e-4)
    part_train.add_argument("--hidden-dim", type=int, default=64)
    part_train.add_argument("--message-passing-layers", type=int, default=2)
    part_train.add_argument("--seed", type=int, default=2601)
    part_train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    part_train.add_argument("--log-every", type=int, default=12)

    part_eval = subparsers.add_parser("part-eval", help="Closed-loop rollout evaluation for oracle part-node task.")
    part_eval.add_argument("--checkpoint-path", required=True)
    part_eval.add_argument("--episodes", type=int, default=64)
    part_eval.add_argument("--max-steps", type=int, default=24)
    part_eval.add_argument("--seed", type=int, default=3601)
    part_eval.add_argument("--output-dir", default="artifacts/part_node_feasibility/eval")
    part_eval.add_argument("--layout", choices=["iid", "ood"], default="iid")
    part_eval.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    part_eval.add_argument("--eval-workers", type=int, default=1)
    part_eval.add_argument("--render-first-episode", action="store_true")

    part_compare = subparsers.add_parser("part-compare", help="Aggregate oracle part-node seed results.")
    part_compare.add_argument("--artifact-dir", default="artifacts/part_node_feasibility")
    part_compare.add_argument("--output-path", default="artifacts/part_node_feasibility/part_node_comparison.json")
    part_compare.add_argument("--summary-path", default="artifacts/part_node_feasibility/summary.md")

    paired = subparsers.add_parser(
        "part-paired-eval",
        help="Run paired closed-loop OOD validation for part_concat vs part_node checkpoints.",
    )
    paired.add_argument("--artifact-dir", default="artifacts/part_node_feasibility")
    paired.add_argument("--output-dir", default="artifacts/part_node_feasibility/paired_ood_validation")
    paired.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    paired.add_argument("--episodes", type=int, default=500)
    paired.add_argument("--max-steps", type=int, default=24)
    paired.add_argument("--layout", choices=["iid", "ood"], default="ood")
    paired.add_argument("--eval-seed-base", type=int, default=3811)
    paired.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    paired.add_argument("--eval-workers", type=int, default=1)
    paired.add_argument("--bootstrap-samples", type=int, default=2000)
    paired.add_argument("--plot-worst", type=int, default=5)
    paired.add_argument("--catastrophic-distance", type=float, default=0.3)

    keypoint = subparsers.add_parser("keypoint-run", help="Run sparse keypoint graph feasibility experiment.")
    keypoint.add_argument("--artifact-dir", default="artifacts/keypoint_graph_feasibility")
    keypoint.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    keypoint.add_argument("--episodes", type=int, default=160)
    keypoint.add_argument("--ood-episodes", type=int, default=80)
    keypoint.add_argument("--eval-episodes", type=int, default=64)
    keypoint.add_argument("--max-steps", type=int, default=24)
    keypoint.add_argument("--epochs", type=int, default=36)
    keypoint.add_argument("--batch-size", type=int, default=256)
    keypoint.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    keypoint.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    keypoint.add_argument("--eval-workers", type=int, default=1)
    keypoint.add_argument("--bootstrap-samples", type=int, default=1000)
    keypoint.add_argument("--plot-worst", type=int, default=3)

    keypoint_diag = subparsers.add_parser(
        "keypoint-diagnostics",
        help="Run sparse keypoint graph sanity checks without changing the main experiment.",
    )
    keypoint_diag.add_argument(
        "--dataset-path",
        default="artifacts/keypoint_graph_feasibility/parallel_mps/seed2811_run/seed2811/dataset/keypoint_alignment.pt",
    )
    keypoint_diag.add_argument(
        "--checkpoint-dir",
        default="artifacts/keypoint_graph_feasibility/parallel_mps/seed2811_run/seed2811/checkpoints",
    )
    keypoint_diag.add_argument("--output-dir", default="artifacts/keypoint_graph_feasibility/diagnostics/seed2811")
    keypoint_diag.add_argument("--seed", type=int, default=2811)
    keypoint_diag.add_argument("--target-epochs", type=int, default=300)
    keypoint_diag.add_argument("--overfit-epochs", type=int, default=500)
    keypoint_diag.add_argument("--overfit-episodes", type=int, default=100)
    keypoint_diag.add_argument("--batch-size", type=int, default=128)
    keypoint_diag.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    keypoint_diag.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    structural = subparsers.add_parser(
        "keypoint-structural-diagnostics",
        help="Run arc-position and EE-readout follow-up diagnostics without overwriting prior artifacts.",
    )
    structural.add_argument("--artifact-dir", default="artifacts/keypoint_graph_feasibility/parallel_mps")
    structural.add_argument("--output-dir", default="artifacts/keypoint_graph_feasibility/structural_diagnostics")
    structural.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    structural.add_argument("--epochs", type=int, default=36)
    structural.add_argument("--target-decode-epochs", type=int, default=300)
    structural.add_argument("--batch-size", type=int, default=256)
    structural.add_argument("--eval-episodes", type=int, default=64)
    structural.add_argument("--max-steps", type=int, default=24)
    structural.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="mps")
    structural.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="mps")
    structural.add_argument("--bootstrap-samples", type=int, default=1000)
    structural.add_argument("--plot-worst", type=int, default=3)

    generalized = subparsers.add_parser(
        "generalized-geometric-run",
        help="Run variable-sampling concat/set/geometric-GNN feasibility experiment.",
    )
    generalized.add_argument("--output-dir", default="artifacts/generalized_geometric_graph_feasibility")
    generalized.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    generalized.add_argument("--train-shapes", type=int, default=48)
    generalized.add_argument("--val-shapes", type=int, default=12)
    generalized.add_argument("--test-shapes", type=int, default=24)
    generalized.add_argument("--eval-shapes", type=int, default=24)
    generalized.add_argument("--max-steps", type=int, default=20)
    generalized.add_argument("--epochs", type=int, default=24)
    generalized.add_argument("--batch-size", type=int, default=128)
    generalized.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    generalized.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    generalized.add_argument("--plot-cases", type=int, default=3)

    orientation = subparsers.add_parser(
        "generalized-orientation-sanity",
        help="Diagnose relative-orientation OOD with existing checkpoints and a widened Set Attention retrain.",
    )
    orientation.add_argument("--source-output-dir", default="artifacts/generalized_geometric_graph_feasibility")
    orientation.add_argument("--output-dir", default="artifacts/generalized_geometric_graph_feasibility/orientation_sanity")
    orientation.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    orientation.add_argument("--wide-epochs", type=int, default=24)
    orientation.add_argument("--batch-size", type=int, default=128)
    orientation.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    orientation.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    orientation.add_argument("--sweep-episodes-per-bin", type=int, default=48)
    orientation.add_argument("--rollout-episodes", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        result = generate_representation_demonstrations(
            RepresentationDemoConfig(
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                output_path=args.output_path,
                metadata_path=args.metadata_path,
                object_x_range=tuple(args.object_x_range),
                object_z_range=tuple(args.object_z_range),
                object_yaw_range=tuple(args.object_yaw_range),
                handle_offset_range=tuple(args.handle_offset_range),
                target_radius=args.target_radius,
                max_delta_ee=args.max_delta_ee,
                camera_yaw=args.camera_yaw,
                crop_size=args.crop_size,
                visual_grid=args.visual_grid,
                render_first_episode=args.render_first_episode,
            )
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "train":
        _, summary = train_representation_policy(
            RepresentationTrainConfig(
                dataset_path=args.dataset_path,
                variant=args.variant,
                model_kind=args.model_kind,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                mpnn_hidden_dim=args.mpnn_hidden_dim,
                mpnn_layers=args.mpnn_layers,
                seed=args.seed,
                device=args.device,
                log_every=args.log_every,
            )
        )
        print(
            json.dumps(
                {
                    "checkpoint_path": summary["checkpoint_path"],
                    "summary_path": summary["summary_path"],
                    "model_kind": summary["model_kind"],
                    "variant": summary["variant"],
                    "offline": summary["offline"],
                    "parameter_count": summary["parameter_count"],
                    "runtime_seconds": summary["runtime_seconds"],
                },
                indent=2,
            )
        )
        return

    if args.command == "eval":
        payload = evaluate_representation_policy(
            RepresentationEvalConfig(
                checkpoint_path=args.checkpoint_path,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                output_dir=args.output_dir,
                device=args.device,
                camera_yaw=args.camera_yaw,
                render_first_episode=args.render_first_episode,
            )
        )
        print(json.dumps({"output_path": payload["output_path"], "summary": payload["summary"]}, indent=2))
        return

    if args.command == "compare":
        print(json.dumps(_comparison_table(Path(args.checkpoint_dir), Path(args.eval_dir)), indent=2))
        return

    if args.command == "compare-architecture":
        payload = _architecture_comparison(
            [Path(path) for path in args.mlp_summaries],
            [Path(path) for path in args.mpnn_summaries],
            [Path(path) for path in args.mlp_evals],
            [Path(path) for path in args.mpnn_evals],
        )
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["output_path"] = str(output_path)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "part-generate":
        result = generate_part_node_demonstrations(
            PartNodeDemoConfig(
                episodes=args.episodes,
                ood_episodes=args.ood_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                output_path=args.output_path,
                metadata_path=args.metadata_path,
                object_x_range=tuple(args.object_x_range),
                object_z_range=tuple(args.object_z_range),
                object_yaw_range=tuple(args.object_yaw_range),
                train_part_local_x_range=tuple(args.train_part_local_x_range),
                train_part_local_z_range=tuple(args.train_part_local_z_range),
                ood_part_local_abs_x_range=tuple(args.ood_part_local_abs_x_range),
                ood_part_local_z_range=tuple(args.ood_part_local_z_range),
                target_radius=args.target_radius,
                max_delta_ee=args.max_delta_ee,
                camera_yaw=args.camera_yaw,
                crop_size=args.crop_size,
                visual_grid=args.visual_grid,
                render_first_episode=args.render_first_episode,
            )
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "part-train":
        _, summary = train_part_node_policy(
            PartNodeTrainConfig(
                dataset_path=args.dataset_path,
                variant=args.variant,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                message_passing_layers=args.message_passing_layers,
                seed=args.seed,
                device=args.device,
                log_every=args.log_every,
            )
        )
        print(
            json.dumps(
                {
                    "checkpoint_path": summary["checkpoint_path"],
                    "summary_path": summary["summary_path"],
                    "variant": summary["variant"],
                    "offline": summary["offline"],
                    "parameter_count": summary["parameter_count"],
                    "runtime_seconds": summary["runtime_seconds"],
                },
                indent=2,
            )
        )
        return

    if args.command == "part-eval":
        payload = evaluate_part_node_policy(
            PartNodeEvalConfig(
                checkpoint_path=args.checkpoint_path,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                output_dir=args.output_dir,
                layout=args.layout,
                eval_device=args.eval_device,
                eval_workers=args.eval_workers,
                render_first_episode=args.render_first_episode,
            )
        )
        print(json.dumps({"output_path": payload["output_path"], "summary": payload["summary"]}, indent=2))
        return

    if args.command == "part-compare":
        payload = _part_node_comparison(Path(args.artifact_dir))
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_part_node_summary_markdown(payload), encoding="utf-8")
        payload["output_path"] = str(output_path)
        payload["summary_path"] = str(summary_path)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "part-paired-eval":
        payload = evaluate_part_node_paired_rollouts(
            PartNodePairedEvalConfig(
                artifact_dir=args.artifact_dir,
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                episodes=args.episodes,
                max_steps=args.max_steps,
                layout=args.layout,
                eval_seed_base=args.eval_seed_base,
                eval_device=args.eval_device,
                eval_workers=args.eval_workers,
                bootstrap_samples=args.bootstrap_samples,
                plot_worst=args.plot_worst,
                catastrophic_distance=args.catastrophic_distance,
            )
        )
        aggregate = payload["aggregate"]
        print(
            json.dumps(
                {
                    "output_path": payload["output_path"],
                    "summary_path": payload["summary_path"],
                    "contingency": aggregate["contingency"],
                    "part_node_only_to_concat_only_ratio": aggregate["part_node_only_to_concat_only_ratio"],
                    "mcnemar_exact_p": aggregate["mcnemar_exact_p"],
                    "success": aggregate["success"],
                    "final_distance_delta": aggregate["final_distance"]["paired_delta_part_node_minus_concat"],
                },
                indent=2,
            )
        )
        return

    if args.command == "keypoint-run":
        payload = run_keypoint_feasibility_experiment(
            KeypointExperimentConfig(
                artifact_dir=args.artifact_dir,
                seeds=tuple(args.seeds),
                episodes=args.episodes,
                ood_episodes=args.ood_episodes,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                eval_device=args.eval_device,
                eval_workers=args.eval_workers,
                bootstrap_samples=args.bootstrap_samples,
                plot_worst=args.plot_worst,
            )
        )
        print(
            json.dumps(
                {
                    "output_path": payload["output_path"],
                    "summary_path": payload["summary_path"],
                    "splits": payload["splits"],
                    "efficiency": payload["efficiency"],
                    "permutation": payload["permutation"],
                },
                indent=2,
            )
        )
        return

    if args.command == "keypoint-diagnostics":
        payload = run_keypoint_diagnostics(
            KeypointDiagnosticConfig(
                dataset_path=args.dataset_path,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                target_epochs=args.target_epochs,
                overfit_epochs=args.overfit_epochs,
                overfit_episodes=args.overfit_episodes,
                batch_size=args.batch_size,
                device=args.device,
                eval_device=args.eval_device,
            )
        )
        print(
            json.dumps(
                {
                    "output_path": payload["output_path"],
                    "summary_path": payload["summary_path"],
                    "target_decode": payload["target_decode"],
                    "action_overfit": payload["action_overfit"],
                    "cpu_mps_consistency": payload["cpu_mps_consistency"],
                },
                indent=2,
            )
        )
        return

    if args.command == "keypoint-structural-diagnostics":
        payload = run_keypoint_structural_diagnostics(
            KeypointStructuralDiagnosticConfig(
                artifact_dir=args.artifact_dir,
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                epochs=args.epochs,
                target_decode_epochs=args.target_decode_epochs,
                batch_size=args.batch_size,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                device=args.device,
                eval_device=args.eval_device,
                bootstrap_samples=args.bootstrap_samples,
                plot_worst=args.plot_worst,
            )
        )
        print(json.dumps({"output_path": payload["output_path"], "summary_path": payload["summary_path"]}, indent=2))
        return

    if args.command == "generalized-geometric-run":
        payload = run_generalized_experiment(
            GeneralizedGeometryConfig(
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                train_shapes=args.train_shapes,
                val_shapes=args.val_shapes,
                test_shapes=args.test_shapes,
                eval_shapes=args.eval_shapes,
                max_steps=args.max_steps,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                eval_device=args.eval_device,
                plot_cases=args.plot_cases,
            )
        )
        print(json.dumps({"output_path": payload["output_path"], "summary_path": payload["summary_path"]}, indent=2))
        return

    if args.command == "generalized-orientation-sanity":
        payload = run_orientation_sanity(
            OrientationSanityConfig(
                source_output_dir=args.source_output_dir,
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                wide_epochs=args.wide_epochs,
                batch_size=args.batch_size,
                device=args.device,
                eval_device=args.eval_device,
                sweep_episodes_per_bin=args.sweep_episodes_per_bin,
                rollout_episodes=args.rollout_episodes,
            )
        )
        print(json.dumps({"output_path": payload["output_path"], "summary_path": payload["summary_path"]}, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


def _comparison_table(checkpoint_dir: Path, eval_dir: Path) -> list[dict[str, object]]:
    rows = []
    for variant in ("pose", "pose_visual", "pose_visual_ee"):
        summary_path = checkpoint_dir / f"{variant}_summary.json"
        eval_path = eval_dir / f"{variant}_normal_eval.json"
        if not summary_path.exists() or not eval_path.exists():
            continue
        train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
        flags = {
            "pose": "yes",
            "visual_feature": "yes" if variant in {"pose_visual", "pose_visual_ee"} else "no",
            "explicit_ee_geometry": "yes" if variant == "pose_visual_ee" else "basic",
        }
        rows.append(
            {
                "variant": variant,
                **flags,
                "test_action_l2_error": train_summary["offline"]["test"]["l2_error"],
                "rollout_success": eval_summary["summary"]["success_rate"],
                "final_distance": eval_summary["summary"]["mean_final_distance"],
                "policy_latency_ms": eval_summary["summary"]["mean_policy_latency_ms"],
                "visual_latency_ms": eval_summary["summary"]["mean_visual_latency_ms"],
                "forward_latency_ms": eval_summary["summary"]["mean_forward_latency_ms"],
            }
        )
    return rows


def _architecture_comparison(
    mlp_summaries: list[Path],
    mpnn_summaries: list[Path],
    mlp_evals: list[Path],
    mpnn_evals: list[Path],
) -> dict[str, object]:
    lengths = {len(mlp_summaries), len(mpnn_summaries), len(mlp_evals), len(mpnn_evals)}
    if len(lengths) != 1:
        raise ValueError("MLP/MPNN summary and eval path lists must have the same length.")
    per_seed = []
    contingency = {"both_success": 0, "mlp_only": 0, "mpnn_only": 0, "both_failure": 0}
    for idx, (mlp_summary_path, mpnn_summary_path, mlp_eval_path, mpnn_eval_path) in enumerate(
        zip(mlp_summaries, mpnn_summaries, mlp_evals, mpnn_evals, strict=True),
        start=1,
    ):
        mlp_summary = json.loads(mlp_summary_path.read_text(encoding="utf-8"))
        mpnn_summary = json.loads(mpnn_summary_path.read_text(encoding="utf-8"))
        mlp_eval = json.loads(mlp_eval_path.read_text(encoding="utf-8"))
        mpnn_eval = json.loads(mpnn_eval_path.read_text(encoding="utf-8"))
        seed_contingency = _paired_contingency(mlp_eval["episodes"], mpnn_eval["episodes"])
        for key, value in seed_contingency.items():
            contingency[key] += value
        per_seed.append(
            {
                "seed_index": idx,
                "mlp": _seed_metrics(mlp_summary, mlp_eval),
                "mpnn": _seed_metrics(mpnn_summary, mpnn_eval),
                "paired_contingency": seed_contingency,
            }
        )
    return {
        "per_seed": per_seed,
        "aggregate": {
            "mlp": _aggregate_metrics([item["mlp"] for item in per_seed]),
            "mpnn": _aggregate_metrics([item["mpnn"] for item in per_seed]),
            "paired_contingency": contingency,
        },
    }


def _seed_metrics(summary: dict[str, object], eval_payload: dict[str, object]) -> dict[str, float]:
    eval_summary = eval_payload["summary"]
    return {
        "params": float(summary["parameter_count"]),
        "action_l2": float(summary["offline"]["test"]["l2_error"]),
        "success": float(eval_summary["success_rate"]),
        "final_distance": float(eval_summary["mean_final_distance"]),
        "policy_latency_ms": float(eval_summary["mean_policy_latency_ms"]),
        "forward_latency_ms": float(eval_summary["mean_forward_latency_ms"]),
    }


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = rows[0].keys()
    return {key: _mean_std([row[key] for row in rows]) for key in keys}


def _part_node_comparison(artifact_dir: Path) -> dict[str, object]:
    variants = ("object_only", "part_concat", "part_node")
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    evals: dict[tuple[str, str, str], dict[str, object]] = {}

    for summary_path in artifact_dir.rglob("*_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "oracle_part_node_feasibility":
            continue
        variant = str(payload["variant"])
        seed_key = _artifact_seed_key(summary_path, payload)
        summaries[(variant, seed_key)] = payload | {"_path": str(summary_path)}

    for eval_path in artifact_dir.rglob("*_eval.json"):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if payload.get("variant") not in variants or payload.get("layout") not in {"iid", "ood"}:
            continue
        variant = str(payload["variant"])
        layout = str(payload["layout"])
        seed_key = _artifact_seed_key(eval_path, payload)
        evals[(variant, seed_key, layout)] = payload | {"_path": str(eval_path)}

    per_seed: list[dict[str, object]] = []
    aggregate: dict[str, object] = {}
    for variant in variants:
        seed_keys = sorted(
            seed
            for (summary_variant, seed) in summaries
            if summary_variant == variant
            and (variant, seed, "iid") in evals
            and (variant, seed, "ood") in evals
        )
        variant_rows: list[dict[str, float]] = []
        for seed_key in seed_keys:
            summary = summaries[(variant, seed_key)]
            iid_eval = evals[(variant, seed_key, "iid")]
            ood_eval = evals[(variant, seed_key, "ood")]
            row = {
                "seed": seed_key,
                "variant": variant,
                "params": float(summary["parameter_count"]),
                "iid_action_l2": float(summary["offline"]["test"]["l2_error"]),
                "ood_action_l2": float(summary["offline"]["ood"]["l2_error"]),
                "iid_success": float(iid_eval["summary"]["success_rate"]),
                "ood_success": float(ood_eval["summary"]["success_rate"]),
                "iid_final_distance": float(iid_eval["summary"]["mean_final_distance"]),
                "ood_final_distance": float(ood_eval["summary"]["mean_final_distance"]),
                "iid_policy_latency_ms": float(iid_eval["summary"]["mean_policy_latency_ms"]),
                "ood_policy_latency_ms": float(ood_eval["summary"]["mean_policy_latency_ms"]),
                "iid_forward_latency_ms": float(iid_eval["summary"]["mean_forward_latency_ms"]),
                "ood_forward_latency_ms": float(ood_eval["summary"]["mean_forward_latency_ms"]),
            }
            per_seed.append(row)
            variant_rows.append({key: value for key, value in row.items() if isinstance(value, float)})
        if variant_rows:
            aggregate[variant] = _aggregate_metrics(variant_rows)

    return {
        "artifact_dir": str(artifact_dir),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "input_difference": {
            "object_only": "Object and EE nodes only; object visual/body pose available; oracle part information withheld.",
            "part_concat": "Object and EE nodes only; the exact oracle part feature vector is concatenated to object features.",
            "part_node": "Object, EE, and Part nodes; the exact oracle part feature vector is used as a separate Part node.",
        },
    }


def _artifact_seed_key(path: Path, payload: dict[str, object]) -> str:
    for part in path.parts:
        if part.startswith("seed"):
            return part
    config = payload.get("config", {})
    if isinstance(config, dict) and "seed" in config:
        return f"seed{config['seed']}"
    return "seed_unknown"


def _part_node_summary_markdown(payload: dict[str, object]) -> str:
    aggregate = payload.get("aggregate", {})
    lines = [
        "# Oracle Part Node Feasibility",
        "",
        "| Model | Params | IID Success | OOD Success | IID Final Dist | OOD Final Dist | IID Action L2 | OOD Action L2 | IID Policy ms | IID Forward ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("object_only", "part_concat", "part_node"):
        metrics = aggregate.get(variant)
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    _fmt_mean_std(metrics["params"], digits=0),
                    _fmt_mean_std(metrics["iid_success"]),
                    _fmt_mean_std(metrics["ood_success"]),
                    _fmt_mean_std(metrics["iid_final_distance"]),
                    _fmt_mean_std(metrics["ood_final_distance"]),
                    _fmt_mean_std(metrics["iid_action_l2"]),
                    _fmt_mean_std(metrics["ood_action_l2"]),
                    _fmt_mean_std(metrics["iid_policy_latency_ms"]),
                    _fmt_mean_std(metrics["iid_forward_latency_ms"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- Part-concat > Object-only: oracle part information itself is useful.",
            "- Part-node > Part-concat: separating the same part information into an entity node adds value.",
            "- Part-node ~= Part-concat: part information matters, but this task does not yet justify a separate part node.",
            "- Part-node < Part-concat: the current relational bias/optimization cost is not paying off.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_mean_std(metric: dict[str, float], digits: int = 4) -> str:
    return f"{metric['mean']:.{digits}f} +/- {metric['std']:.{digits}f}"


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_contingency(mlp_episodes: list[dict[str, object]], mpnn_episodes: list[dict[str, object]]) -> dict[str, int]:
    if len(mlp_episodes) != len(mpnn_episodes):
        raise ValueError("Paired eval files must contain the same number of episodes.")
    counts = {"both_success": 0, "mlp_only": 0, "mpnn_only": 0, "both_failure": 0}
    for mlp_ep, mpnn_ep in zip(mlp_episodes, mpnn_episodes, strict=True):
        mlp_success = bool(mlp_ep["success"])
        mpnn_success = bool(mpnn_ep["success"])
        if mlp_success and mpnn_success:
            counts["both_success"] += 1
        elif mlp_success and not mpnn_success:
            counts["mlp_only"] += 1
        elif not mlp_success and mpnn_success:
            counts["mpnn_only"] += 1
        else:
            counts["both_failure"] += 1
    return counts


if __name__ == "__main__":
    main()
