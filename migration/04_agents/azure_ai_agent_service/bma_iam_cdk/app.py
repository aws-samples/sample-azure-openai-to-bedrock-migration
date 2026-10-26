#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Least-privilege IAM for the Bedrock Managed Agents (preview) researcher example.

Creates exactly two IAM roles — no compute, no storage, no cost:

  BmaClientRole   What YOUR code assumes to call the BMA API (and what the Codex
                  exec server signs with). Session CRUD + environment attach,
                  plus iam:PassRole for the session role ONLY, and only to
                  bedrock-mantle.amazonaws.com.

  SessionRole     What BMA itself assumes (the `role_arn` you pass when creating
                  a session). Its only permission is bedrock-mantle:CreateInference
                  for the models listed in context `models` (default:
                  openai.gpt-5.6-luna). The bundle AWS ships for the preview
                  grants CreateInference on every model, plus Web Search; the BMA
                  docs say to narrow that before real use, and this does.

The session role keeps the bundle's fixed name so AWS's example scripts work
unchanged against it. Only one stack can own a role name: deploying AWS's
`self-hosted` example into the same account needs ManageCustomerInferenceRole=false.

Deploy / destroy (from this directory):

    pip install -r requirements.txt
    cdk deploy                       # or: cdk deploy -c models=openai.gpt-5.6-luna,openai.gpt-5.6-terra
    cdk destroy

Authorization is Azure's `DefaultAzureCredential` -> AWS SigV4: callers sign with
service `bedrock-mantle` using these roles; no API key is involved.
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from aws_cdk import Aspects, Stack, aws_iam as iam
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from constructs import Construct

BMA_SERVICE_PRINCIPAL = "bedrock-mantle.amazonaws.com"
SESSION_ROLE_NAME = "BedrockManagedAgentsPreviewInferenceServiceRole"
DEFAULT_MODELS = ["openai.gpt-5.6-luna"]

# API actions the client identity needs (names are the preview's, per the BMA
# security page). RegisterEnvironment/ConnectEnvironment are for the exec server
# that attaches a self-hosted environment.
SESSION_ACTIONS = [
    "bedrock-mantle:CreateAgentSession",
    "bedrock-mantle:GetAgentSession",
    "bedrock-mantle:ListAgentSessions",
    "bedrock-mantle:DeleteAgentSession",
    "bedrock-mantle:CreateAgentSessionEvent",
    "bedrock-mantle:ListAgentSessionEvents",
    "bedrock-mantle:ListAgentSessionItems",
]
ENVIRONMENT_ACTIONS = [
    "bedrock-mantle:RegisterEnvironment",
    "bedrock-mantle:ConnectEnvironment",
]


class BmaIamStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, models: list[str], **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Session role: assumed by BMA, inference on named models only ──────
        session_role = iam.Role(
            self,
            "SessionRole",
            role_name=SESSION_ROLE_NAME,
            description="Assumed by Bedrock Managed Agents (preview) for model inference",
            max_session_duration=cdk.Duration.hours(1),
            # Confused-deputy protection: only bedrock-mantle acting for THIS account.
            assumed_by=iam.ServicePrincipal(
                BMA_SERVICE_PRINCIPAL,
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:{self.partition}:bedrock-mantle:*:{self.account}:*"
                    },
                },
            ),
        )
        session_role.add_to_policy(
            iam.PolicyStatement(
                sid="CreateInferenceForApprovedModelsOnly",
                actions=["bedrock-mantle:CreateInference"],
                # CreateInference has no model resource ARN; the model condition
                # is what narrows it (BMA docs, "Configure the session role").
                resources=["*"],
                conditions={"StringEquals": {"bedrock-mantle:Model": models}},
            )
        )

        # ── Client role: what the caller (and the exec server) signs with ─────
        client_role = iam.Role(
            self,
            "ClientRole",
            role_name="BmaClientRole",
            description="Client identity for the BMA researcher example",
            max_session_duration=cdk.Duration.hours(1),
            # Same-account principals that are ALSO allowed sts:AssumeRole on it.
            assumed_by=iam.AccountPrincipal(self.account),
        )
        project_arn = self.format_arn(
            service="bedrock-mantle", region="*", account=self.account,
            resource="project", resource_name="*",
        )
        client_role.add_to_policy(
            iam.PolicyStatement(
                sid="UseBmaSessions",
                actions=SESSION_ACTIONS + ENVIRONMENT_ACTIONS,
                resources=[project_arn],
            )
        )
        client_role.add_to_policy(
            iam.PolicyStatement(
                sid="PassOnlyTheSessionRoleToBma",
                actions=["iam:PassRole"],
                resources=[session_role.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": BMA_SERVICE_PRINCIPAL}},
            )
        )

        # ── cdk-nag: the two wildcards are service limitations, not laziness ──
        NagSuppressions.add_resource_suppressions(
            session_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "bedrock-mantle:CreateInference does not support a model resource ARN; "
                        "it is constrained by the bedrock-mantle:Model condition to the models "
                        "in context `models`."
                    ),
                    "appliesTo": ["Resource::*"],
                }
            ],
            apply_to_children=True,
        )
        NagSuppressions.add_resource_suppressions(
            client_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "BMA preview authorizes against project ARNs "
                        "(arn:aws:bedrock-mantle:*:<account>:project/*); the project id is "
                        "service-assigned, so the name segment cannot be enumerated in advance. "
                        "Scoped to this account."
                    ),
                    "appliesTo": [{"regex": "/^Resource::arn:.*:bedrock-mantle:\\*:.*:project/\\*$/"}],
                }
            ],
            apply_to_children=True,
        )

        cdk.CfnOutput(self, "ClientRoleArn", value=client_role.role_arn)
        cdk.CfnOutput(self, "SessionRoleArn", value=session_role.role_arn)


app = cdk.App()
_models = [m.strip() for m in str(app.node.try_get_context("models") or "").split(",") if m.strip()]
BmaIamStack(
    app,
    "BmaResearcherIamStack",
    models=_models or DEFAULT_MODELS,
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
)
Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
app.synth()
