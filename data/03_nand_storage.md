# NAND Flash and SSD Storage

## What is NAND flash?

NAND flash is a type of non-volatile memory, meaning it retains data even when power is
switched off. It is the storage technology inside solid-state drives (SSDs), USB flash
drives, and memory cards. Data is stored in memory cells made of floating-gate
transistors. NAND flash is organized into pages and blocks: data is written and read at
the page level, but it can only be erased at the larger block level. This asymmetry is a
defining characteristic of how flash storage behaves.

## Cell types: SLC, MLC, TLC, QLC

NAND cells are classified by how many bits each cell stores. Single-Level Cell (SLC)
stores one bit per cell and offers the highest speed and endurance but the lowest
density and highest cost. Multi-Level Cell (MLC) stores two bits, Triple-Level Cell
(TLC) stores three bits, and Quad-Level Cell (QLC) stores four bits. As the number of
bits per cell increases, storage density and cost-efficiency improve, but write speed
and endurance decrease. Choosing a cell type is a trade-off between cost, capacity,
performance, and lifespan.

## The role of the controller

An SSD is not just NAND chips; it also contains a controller, which is a specialized
processor that manages the flash. The controller handles wear leveling (spreading writes
evenly across cells so no single block wears out early), error correction, garbage
collection (reclaiming space from deleted data), and the mapping between logical
addresses seen by the computer and the physical locations in flash. The quality of the
controller and its firmware has a large impact on an SSD's real-world performance and
reliability. Companies that design these controllers, such as Phison, sit at the heart
of the storage industry.

## Endurance and wear

Flash cells wear out after a finite number of program/erase cycles. Endurance is often
measured in terabytes written (TBW) or drive writes per day (DWPD). Enterprise and data
center workloads demand high endurance, which is one reason data center SSDs may use
more durable cell types and over-provision extra capacity. Managing endurance well is a
key engineering challenge in storage design.

## Storage for AI workloads

AI training and inference place heavy demands on storage. Large language models require
fast access to enormous datasets and model checkpoints. On-premise storage solutions can
keep sensitive data in-house for security and sovereignty reasons, while still providing
the throughput that AI workloads need. This intersection of storage and AI is an active
area of innovation in the semiconductor industry.
