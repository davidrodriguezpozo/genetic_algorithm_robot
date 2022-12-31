import copy
import math
import re
from typing import Dict, List
from xml.dom.minidom import Document, Node

import numpy as np

Gene = np.ndarray
Genome = List[np.ndarray]


class URDFLink:
    def __init__(
        self,
        name: str,
        parent_name: str,
        recur: int,
        link_shape=0.1,
        link_length=0.1,
        link_radius=0.1,
        link_recurrence=0.1,
        link_mass=0.1,
        joint_type=0.1,
        joint_parent=0.1,
        joint_axis_xyz=0.1,
        joint_origin_rpy_1=0.1,
        joint_origin_rpy_2=0.1,
        joint_origin_rpy_3=0.1,
        joint_origin_xyz_1=0.1,
        joint_origin_xyz_2=0.1,
        joint_origin_xyz_3=0.1,
        control_waveform=0.1,
        control_amp=0.1,
        control_freq=0.1,
    ):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur
        self.link_shape = link_shape
        self.link_length = link_length
        self.link_radius = link_radius
        self.link_recurrence = link_recurrence
        self.link_mass = link_mass
        self.joint_type = joint_type
        self.joint_parent = joint_parent
        self.joint_axis_xyz = joint_axis_xyz
        self.joint_origin_rpy_1 = joint_origin_rpy_1
        self.joint_origin_rpy_2 = joint_origin_rpy_2
        self.joint_origin_rpy_3 = joint_origin_rpy_3
        self.joint_origin_xyz_1 = joint_origin_xyz_1
        self.joint_origin_xyz_2 = joint_origin_xyz_2
        self.joint_origin_xyz_3 = joint_origin_xyz_3
        self.control_waveform = control_waveform
        self.control_amp = control_amp
        self.control_freq = control_freq
        self.sibling_ind = 1

    def to_link_element(self, adom):
        #         <link name="base_link">
        #     <visual>
        #       <geometry>
        #         <cylinder length="0.6" radius="0.25"/>
        #       </geometry>
        #     </visual>
        #     <collision>
        #       <geometry>
        #         <cylinder length="0.6" radius="0.25"/>
        #       </geometry>
        #     </collision>
        #     <inertial>
        # 	    <mass value="0.25"/>
        # 	    <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        #     </inertial>
        #   </link>

        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        cyl_tag = adom.createElement("cylinder")
        cyl_tag.setAttribute("length", str(self.link_length))
        cyl_tag.setAttribute("radius", str(self.link_radius))

        geom_tag.appendChild(cyl_tag)
        vis_tag.appendChild(geom_tag)

        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        c_cyl_tag = adom.createElement("cylinder")
        c_cyl_tag.setAttribute("length", str(self.link_length))
        c_cyl_tag.setAttribute("radius", str(self.link_radius))

        c_geom_tag.appendChild(c_cyl_tag)
        coll_tag.appendChild(c_geom_tag)

        #     <inertial>
        # 	    <mass value="0.25"/>
        # 	    <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        #     </inertial>
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        # pi r^2 * height
        mass = np.pi * (self.link_radius * self.link_radius) * self.link_length
        mass_tag.setAttribute("value", str(mass))
        inertia_tag = adom.createElement("inertia")
        # <inertia ixx="0.0003" iyy="0.0003" izz="0.0003" ixy="0" ixz="0" iyz="0"/>
        inertia_tag.setAttribute("ixx", "0.03")
        inertia_tag.setAttribute("iyy", "0.03")
        inertia_tag.setAttribute("izz", "0.03")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)

        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)

        return link_tag

    def to_joint_element(self, adom):
        #           <joint name="base_to_sub2" type="revolute">
        #     <parent link="base_link"/>
        #     <child link="sub_link2"/>
        #     <axis xyz="1 0 0"/>
        #     <limit effort="10" upper="0" lower="10" velocity="1"/>
        #     <origin rpy="0 0 0" xyz="0 0.5 0"/>
        #   </joint>
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        if self.joint_type >= 0.5:
            joint_tag.setAttribute("type", "revolute")
        else:
            joint_tag.setAttribute("type", "revolute")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        if self.joint_axis_xyz > 0.33 and self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        if self.joint_axis_xyz > 0.66:
            axis_tag.setAttribute("xyz", "0 0 1")

        limit_tag = adom.createElement("limit")
        # effort upper lower velocity
        limit_tag.setAttribute("effort", "1")
        limit_tag.setAttribute("upper", "-3.1415")
        limit_tag.setAttribute("lower", "3.1415")
        limit_tag.setAttribute("velocity", "1")
        # <origin rpy="0 0 0" xyz="0 0.5 0"/>
        orig_tag = adom.createElement("origin")

        rpy1 = self.joint_origin_rpy_1 * self.sibling_ind

        rpy = (
            str(rpy1)
            + " "
            + str(self.joint_origin_rpy_2)
            + " "
            + str(self.joint_origin_rpy_3)
        )
        orig_tag.setAttribute("rpy", rpy)
        xyz = (
            str(self.joint_origin_xyz_1)
            + " "
            + str(self.joint_origin_xyz_2)
            + " "
            + str(self.joint_origin_xyz_3)
        )
        orig_tag.setAttribute("xyz", xyz)

        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)
        return joint_tag


class GenomeGenerator:
    @staticmethod
    def get_random_gene(n: int) -> Gene:
        gene = [np.random.random() for _ in range(n)]
        return gene

    @staticmethod
    def get_random_genome(gene_length: int, gene_count: int) -> Genome:
        genome = [
            GenomeGenerator.get_random_gene(gene_length) for _ in range(gene_count)
        ]
        return genome

    @staticmethod
    def get_gene_spec() -> Dict:
        gene_spec = {
            "link-shape": {"scale": 1},
            "link-length": {"scale": 1},
            "link-radius": {"scale": 1},
            "link-recurrence": {"scale": 4},
            "link-mass": {"scale": 1},
            "joint-type": {"scale": 1},
            "joint-parent": {"scale": 1},
            "joint-axis-xyz": {"scale": 1},
            "joint-origin-rpy-1": {"scale": np.pi * 2},
            "joint-origin-rpy-2": {"scale": np.pi * 2},
            "joint-origin-rpy-3": {"scale": np.pi * 2},
            "joint-origin-xyz-1": {"scale": 1},
            "joint-origin-xyz-2": {"scale": 1},
            "joint-origin-xyz-3": {"scale": 1},
            "control-waveform": {"scale": 1},
            "control-amp": {"scale": 0.25},
            "control-freq": {"scale": 1},
        }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
        ind = ind + 1
        return gene_spec

    @staticmethod
    def get_gene_dict(gene: np.ndarray, spec: Dict):
        mapped_gene = dict()
        for k in spec:
            mapped_gene[k] = gene[spec[k]["ind"]] * spec[k]["scale"]
        return mapped_gene

    @staticmethod
    def get_genome_dicts(genome: Genome, spec: Dict) -> List[Dict]:
        mapped_dicts = []
        for gene in genome:
            mapped_dicts.append(GenomeGenerator.get_gene_dict(gene, spec))
        return mapped_dicts

    @staticmethod
    def to_snake(s: str):
        return re.sub("([-])", "_", s).lower()

    @staticmethod
    def genome_to_links(gdicts):
        links = []
        link_ind = 0
        parent_names = [str(link_ind)]
        for gdict in gdicts:
            link_name = str(link_ind)
            parent_ind = gdict["joint-parent"] * (len(parent_names) - 1)
            parent_name = parent_names[int(parent_ind)]
            # print("available parents: ", parent_names, "chose", parent_name)
            recur = gdict["link-recurrence"]
            link = URDFLink(
                name=link_name,
                parent_name=parent_name,
                recur=recur + 1,
                link_length=gdict["link-length"],
                link_radius=gdict["link-radius"],
                link_mass=gdict["link-mass"],
                joint_type=gdict["joint-type"],
                joint_parent=gdict["joint-parent"],
                joint_axis_xyz=gdict["joint-axis-xyz"],
                joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                control_waveform=gdict["control-waveform"],
                control_amp=gdict["control-amp"],
                control_freq=gdict["control-freq"],
            )
            links.append(link)
            if link_ind != 0:  # don't re-add the first link
                parent_names.append(link_name)
            link_ind = link_ind + 1

        # now just fix the first link so it links to nothing
        links[0].parent_name = "None"
        return links

    @staticmethod
    def get_children_for_parent_link(
        parent_link: URDFLink, flat_links: List[URDFLink]
    ) -> List[URDFLink]:
        return [link for link in flat_links if link.parent_name == parent_link.name]

    @staticmethod
    def expand_links(
        flat_links: List[URDFLink],
        parent_link: URDFLink = None,
        uniq_parent_name: str = None,
        exp_links: List[URDFLink] = None,
    ):
        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1
        for c in children:
            for _ in range(int(c.recur)):
                sibling_ind = sibling_ind + 1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                # print("exp: ", c.name, " -> ", uniq_name)
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                exp_links.append(c_copy)
                assert c.parent_name != c.name, (
                    "Genome::expandLinks: link joined to itself: "
                    + c.name
                    + " joins "
                    + c.parent_name
                )
                GenomeGenerator.expand_links(flat_links, c, uniq_name, exp_links)

    @staticmethod
    def crossover(g1, g2):
        xo = np.random.randint(len(g1))
        if xo > len(g2):
            xo = len(g2) - 1
        if xo == 0:
            xo = 1
        g3 = np.concatenate((g1[0:xo], g2[xo:]))
        return g3

    @staticmethod
    def point_mutate(genome, rate, amount):
        for gene in genome:
            if np.random.rand() < rate:
                ind = np.random.randint(len(gene))
                r = (np.random.rand() - 0.5) * amount
                gene[ind] = gene[ind] + r
        return genome

    @staticmethod
    def grow_mutate(genome, rate):
        if np.random.rand() < rate:
            g = GenomeGenerator.get_random_gene(len(genome[0]))
            np.append(genome, [g], axis=0)
        return genome

    @staticmethod
    def shrink_mutate(genome, rate):
        genes = genome
        if len(genome) > 1 and np.random.rand() < rate:
            ind = np.random.randint(len(genome))
            genes = np.delete(genome, ind, 0)
        return genes

    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + "\n"
        with open(csv_file, "w") as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(csv_file):
        with open(csv_file) as f:
            content = f.read()
            lines = content.split("\n")
            dna = []
            for line in lines:
                vals = line.split(",")
                gene = [float(v) for v in vals if v != ""]
                if len(gene):
                    dna.append(gene)
            return dna
